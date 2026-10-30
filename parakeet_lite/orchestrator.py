from __future__ import annotations

import threading
from collections.abc import Callable, Generator
from pathlib import Path

from .audio import AudioRecorder
from .config import ConfigManager
from .llm import LLMClient
from .model_manager import ModelManager
from .models import AppConfig, AppState, SystemStatus, TranscriptionResult
from .transcriber import Transcriber
from .web_search import (
    DuckDuckGoSearch,
    SearchDiscriminator,
    format_search_context,
)


AI_DEVELOPER_TERMINOLOGY = """
When the interview domain is Technology and the context suggests AI, software, or
machine learning, use these canonical spellings. Correct likely speech-to-text
variants to the canonical spelling, but only when the surrounding sentence supports
the term:
- LangChain (not "long chain" or "lang chain")
- LangGraph (not "long graph" or "lang graph")
- LangSmith (not "long smith" or "lang smith")
- Hugging Face (not "hugging face" as a technical product name variant)
- OpenAI, ChatGPT, GPT, LLM, RAG, and NLP
- Python, JavaScript, TypeScript, FastAPI, Django, React, and Node.js
- PyTorch, TensorFlow, scikit-learn, Transformers, and BERT
- PostgreSQL, MongoDB, Redis, Docker, Kubernetes, and GitHub
- AWS, Amazon Bedrock, Lambda, S3, Azure, and Google Cloud
- embeddings, vector database, vector store, prompt engineering, agents,
  tool calling, function calling, fine-tuning, evaluation, and observability
Never insert a glossary term merely because it appears here. Use it only to repair
a plausible recognition error or to standardize a term already implied by the text.
"""

DOMAIN_TERMINOLOGY = {
    "management": """
Use these canonical management terms only when the context supports them:
stakeholder, roadmap, OKR, KPI, strategy, prioritization, scope, deliverable,
milestone, risk management, dependency, cross-functional, alignment, delegation,
performance review, change management, project management, Agile, Scrum, and
continuous improvement.
""",
    "design": """
Use these canonical design terms only when the context supports them:
user experience, UX, user interface, UI, usability, accessibility, persona,
user journey, customer journey, wireframe, prototype, design system, component,
layout, typography, color palette, interaction design, visual hierarchy,
information architecture, user research, usability testing, and Figma.
""",
    "finance": """
Use these canonical finance terms only when the context supports them:
revenue, gross margin, net income, EBITDA, cash flow, balance sheet, income
statement, forecast, budget, variance, accounts payable, accounts receivable,
ROI, audit, compliance, risk assessment, assets, liabilities, equity,
depreciation, and financial modeling.
""",
    "marketing": """
Use these canonical marketing terms only when the context supports them:
brand awareness, target audience, buyer persona, customer acquisition cost,
customer lifetime value, conversion rate, click-through rate, engagement,
impressions, reach, funnel, lead generation, content strategy, SEO, SEM,
campaign, A/B testing, attribution, and marketing qualified lead.
""",
    "operations": """
Use these canonical operations terms only when the context supports them:
process improvement, standard operating procedure, SOP, workflow, throughput,
capacity planning, supply chain, inventory, procurement, quality assurance,
quality control, SLA, logistics, root cause analysis, lean, Six Sigma,
continuous improvement, and operational efficiency.
""",
    "sales": """
Use these canonical sales terms only when the context supports them:
prospect, lead, qualification, discovery call, sales pipeline, opportunity,
account executive, account manager, value proposition, objection handling,
negotiation, proposal, closing, quota, territory, CRM, conversion rate,
customer relationship, and sales enablement.
""",
    "human resources": """
Use these canonical HR terms only when the context supports them:
recruitment, talent acquisition, onboarding, employee engagement, performance
management, compensation, benefits, payroll, policy, employee relations,
learning and development, succession planning, diversity and inclusion,
workforce planning, applicant tracking system, and compliance.
""",
    "customer support": """
Use these canonical customer-support terms only when the context supports them:
ticket, case, escalation, SLA, first response time, resolution time, knowledge
base, troubleshooting, customer satisfaction, CSAT, NPS, service level,
incident, queue, root cause, and customer experience.
""",
}


class Orchestrator:
    def __init__(
        self,
        config: AppConfig,
        level_callback: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config
        self.model_manager = ModelManager()
        self.llm = LLMClient(
            config.llm.base_url, config.llm.default_model, config.llm.timeout
        )
        self.transcriber = Transcriber(self.model_manager, config.transcription.model_name)
        self.web_search = DuckDuckGoSearch(max_results=5)
        self.search_discriminator = SearchDiscriminator()
        self.recorder = AudioRecorder(
            config.audio.sample_rate,
            config.audio.channels,
            config.audio.chunk_size,
            config.audio.max_duration,
            level_callback,
        )
        self.status = SystemStatus(gpu=self.model_manager.gpu)
        self._lock = threading.Lock()

    def initialize(self) -> bool:
        connected = self.llm.check_connectivity()
        with self._lock:
            self.status.ollama_connected = connected
            self.status.message = "Ollama connected" if connected else "Ollama is not running"
        return connected

    def start_recording(self) -> bool:
        try:
            result = self.recorder.start_recording()
        except RuntimeError as exc:
            self._set_error(str(exc))
            raise
        with self._lock:
            self.status.state = AppState.RECORDING
            self.status.message = "Recording..."
        return result

    def stop_recording(self) -> bytes | None:
        data = self.recorder.stop_recording()
        with self._lock:
            self.status.state = AppState.IDLE
            self.status.message = "Recording stopped"
        return data

    def transcribe_audio(self, audio_data: bytes) -> TranscriptionResult:
        with self._lock:
            self.status.state = AppState.TRANSCRIBING
            self.status.message = "Transcribing..."
        result = self.transcriber.transcribe_audio(audio_data, self.config.audio.sample_rate)
        with self._lock:
            self.status.state = AppState.ERROR if result.error else AppState.IDLE
            self.status.message = result.error or "Transcription complete"
        return result

    def generate_response(self, transcription: str) -> Generator[str, None, None]:
        with self._lock:
            self.status.state = AppState.GENERATING
            self.status.message = "Generating answer..."
        try:
            self.llm.set_model_parameters(
                temperature=self.config.llm.temperature,
                top_p=self.config.llm.top_p,
                num_predict=self.config.llm.max_tokens,
            )
            yield from self._generate_llm(transcription, allow_web_search=True)
        except Exception as exc:
            self._set_error(str(exc))
            raise
        else:
            with self._lock:
                self.status.state = AppState.IDLE
                self.status.message = "Answer complete"

    def improve_transcription(
        self, transcription: str, interview_role: str, interview_domain: str
    ) -> str:
        role = interview_role.strip() or "the interview role"
        domain = interview_domain.strip() or "General"
        prompt = (
            "You are correcting speech-to-text from an interview. Return only the corrected "
            "transcription, with no explanation, quotation marks, or headings.\n"
            f"Interview role: {role}\n"
            f"Interview domain: {domain}\n"
            "Prefer vocabulary appropriate to this domain when the surrounding context "
            "supports it. Do not add specialized terms that are not supported by the text.\n"
            + (
                AI_DEVELOPER_TERMINOLOGY
                if domain.lower() == "technology"
                else DOMAIN_TERMINOLOGY.get(domain.lower(), "")
            )
            + (
                "For a technology interview, preserve product and framework names exactly "
                "as written in the glossary when they are present or clearly implied.\n"
                if domain.lower() == "technology"
                else ""
            )
            + "Fix punctuation, grammar, obvious homophones, and likely transcription mistakes. "
            "Preserve the speaker's meaning and do not invent facts.\n\n"
            + f"Transcription:\n{transcription.strip()}"
        )
        return "".join(self._generate_llm(prompt, allow_web_search=False))

    def _generate_llm(self, prompt: str, allow_web_search: bool) -> Generator[str, None, None]:
        decision = self.search_discriminator.decide(prompt)
        if allow_web_search and decision.should_search:
            with self._lock:
                self.status.message = "Searching the web for current information..."
            results = self.web_search.search(prompt)
            context = format_search_context(results)
            if context:
                prompt = (
                    "Answer the user's question using the web sources below. Cite sources "
                    "as [1], [2], etc. Do not claim a source says something it does not.\n\n"
                    f"{context}\n\nUser question:\n{prompt}"
                )
        yield from self.llm.generate_response(
            prompt, self.config.llm.default_model, self.config.llm.stream
        )

    def get_status(self) -> SystemStatus:
        with self._lock:
            return SystemStatus(
                self.status.state,
                self.status.gpu,
                self.status.ollama_connected,
                self.status.message,
                list(self.status.active_operations),
            )

    def shutdown(self, force: bool = False) -> bool:
        if self.recorder.is_recording():
            if not force:
                return False
            self.recorder.stop_recording()
        self.transcriber.unload_model()
        with self._lock:
            self.status.state = AppState.IDLE
            self.status.message = "Shutdown complete"
        return True

    def _set_error(self, message: str) -> None:
        with self._lock:
            self.status.state = AppState.ERROR
            self.status.message = message


def load_or_create_config(path: str | Path = "config.json") -> AppConfig:
    return ConfigManager(path).load_config()
