from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk

from .models import AppState
from .orchestrator import Orchestrator


class UIController:
    COLORS = {
        "background": "#111827",
        "surface": "#1f2937",
        "surface_alt": "#273449",
        "border": "#374151",
        "text": "#f9fafb",
        "muted": "#9ca3af",
        "accent": "#38bdf8",
        "accent_dark": "#0284c7",
        "success": "#34d399",
        "danger": "#fb7185",
        "input": "#0f172a",
    }

    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator
        config = orchestrator.config
        self.root = tk.Tk()
        self.root.title(config.ui.window_title)
        self.root.geometry(f"{config.ui.window_size[0]}x{config.ui.window_size[1]}")
        self.root.minsize(720, 540)
        self.root.configure(bg=self.COLORS["background"])
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        self.status_var = tk.StringVar(value="Starting...")
        self.status_detail_var = tk.StringVar(value="Preparing local services")
        self.gpu_var = tk.StringVar(value="GPU: checking")
        self.connection_var = tk.StringVar(value="Ollama: checking")
        self.level_var = tk.DoubleVar(value=0.0)
        self.timer_var = tk.StringVar(value="00:00 / 01:00")
        self.interview_role_var = tk.StringVar()
        self.interview_domain_var = tk.StringVar(value="General")
        self._recording_started = 0.0
        self._poll_id: str | None = None
        self._build_style()
        self._build_widgets()
        orchestrator.recorder.level_callback = self.update_audio_level
        if config.hotkeys.enabled:
            self.root.bind(config.hotkeys.record, lambda _: self.start_recording())
            self.root.bind(config.hotkeys.stop, lambda _: self.stop_recording())
            self.root.bind(config.hotkeys.answer, lambda _: self.suggest_answer())
        self.root.after(100, self._poll_status)

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=self.COLORS["background"], foreground=self.COLORS["text"])
        style.configure("Surface.TFrame", background=self.COLORS["surface"])
        style.configure("Toolbar.TFrame", background=self.COLORS["surface"])
        style.configure("Title.TLabel", background=self.COLORS["background"],
                        foreground=self.COLORS["text"], font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", background=self.COLORS["background"],
                        foreground=self.COLORS["muted"], font=("Segoe UI", 9))
        style.configure("Surface.TLabel", background=self.COLORS["surface"],
                        foreground=self.COLORS["text"], font=("Segoe UI", 9))
        style.configure("Muted.Surface.TLabel", background=self.COLORS["surface"],
                        foreground=self.COLORS["muted"], font=("Segoe UI", 9))
        style.configure("Section.TLabel", background=self.COLORS["surface"],
                        foreground=self.COLORS["text"], font=("Segoe UI", 11, "bold"))
        style.configure("Primary.TButton", background=self.COLORS["accent_dark"],
                        foreground="#ffffff", padding=(14, 8), font=("Segoe UI", 10, "bold"))
        style.map("Primary.TButton", background=[("active", self.COLORS["accent"])])
        style.configure("Danger.TButton", background="#9f1239", foreground="#ffffff",
                        padding=(14, 8), font=("Segoe UI", 10, "bold"))
        style.map("Danger.TButton", background=[("active", self.COLORS["danger"])])
        style.configure("Secondary.TButton", background=self.COLORS["surface_alt"],
                        foreground=self.COLORS["text"], padding=(12, 8))
        style.map("Secondary.TButton", background=[("active", self.COLORS["border"])])
        style.configure("Status.TLabel", background=self.COLORS["surface"],
                        foreground=self.COLORS["muted"], padding=(12, 8))
        style.configure("Audio.Horizontal.TProgressbar", troughcolor=self.COLORS["input"],
                        background=self.COLORS["accent"], lightcolor=self.COLORS["accent"],
                        darkcolor=self.COLORS["accent"])
        style.configure("Work.Horizontal.TProgressbar", troughcolor=self.COLORS["input"],
                        background=self.COLORS["accent"], lightcolor=self.COLORS["accent"],
                        darkcolor=self.COLORS["accent"])
        style.configure(
            "Context.TEntry",
            fieldbackground=self.COLORS["input"],
            foreground=self.COLORS["text"],
            insertcolor=self.COLORS["accent"],
            bordercolor=self.COLORS["border"],
            lightcolor=self.COLORS["border"],
            darkcolor=self.COLORS["border"],
            padding=(8, 5),
        )
        style.configure(
            "Context.TCombobox",
            fieldbackground=self.COLORS["input"],
            foreground=self.COLORS["text"],
            background=self.COLORS["input"],
            arrowcolor=self.COLORS["text"],
            bordercolor=self.COLORS["border"],
            lightcolor=self.COLORS["border"],
            darkcolor=self.COLORS["border"],
            padding=(6, 4),
        )
        self.root.option_add("*TCombobox*Listbox.background", self.COLORS["input"])
        self.root.option_add("*TCombobox*Listbox.foreground", self.COLORS["text"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", self.COLORS["accent_dark"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

    def _build_widgets(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        header = ttk.Frame(self.root, padding=(24, 20, 24, 12))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Parakeet Lite", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header, text="Private voice-to-answer assistant running on your computer",
            style="Subtitle.TLabel"
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))
        badges = ttk.Frame(header)
        badges.grid(row=0, column=1, rowspan=2, sticky="e")
        self.connection_badge = ttk.Label(badges, textvariable=self.connection_var,
                                          style="Status.TLabel")
        self.connection_badge.pack(side="left", padx=(0, 8))
        self.gpu_badge = ttk.Label(badges, textvariable=self.gpu_var, style="Status.TLabel")
        self.gpu_badge.pack(side="left")

        controls = ttk.Frame(self.root, style="Toolbar.TFrame", padding=14)
        controls.grid(row=1, column=0, sticky="ew", padx=24, pady=(0, 14))
        controls.columnconfigure(3, weight=1)
        self.start_button = ttk.Button(
            controls, text="Start recording", command=self.start_recording, style="Primary.TButton"
        )
        self.start_button.grid(row=0, column=0, padx=(0, 8))
        self.stop_button = ttk.Button(
            controls, text="Stop", command=self.stop_recording, style="Danger.TButton"
        )
        self.stop_button.grid(row=0, column=1, padx=(0, 14))
        ttk.Label(controls, text="Input level", style="Muted.Surface.TLabel").grid(
            row=0, column=2, padx=(0, 8)
        )
        self.level = ttk.Progressbar(
            controls, variable=self.level_var, maximum=1.0,
            style="Audio.Horizontal.TProgressbar"
        )
        self.level.grid(row=0, column=3, sticky="ew", padx=(0, 12))
        ttk.Label(controls, textvariable=self.timer_var, style="Muted.Surface.TLabel").grid(
            row=0, column=4
        )

        content = ttk.PanedWindow(self.root, orient="horizontal")
        content.grid(row=2, column=0, sticky="nsew", padx=24, pady=(0, 14))
        transcription_card = self._create_editor_card(
            content, "Your prompt", "Edit the transcription before asking the local AI."
        )
        self.transcription = transcription_card[0]
        content.add(transcription_card[1], weight=1)
        response_card = self._create_editor_card(
            content, "Assistant response", "Streamed from your local Ollama model."
        )
        self.response = response_card[0]
        self.response.configure(state="disabled")
        button_bar = response_card[2]
        ttk.Button(button_bar, text="Copy response", command=self._copy_response,
                   style="Secondary.TButton").pack(side="left")
        ttk.Button(button_bar, text="Clear", command=self._clear_response,
                   style="Secondary.TButton").pack(side="left", padx=8)
        content.add(response_card[1], weight=2)

        footer = ttk.Frame(self.root, style="Surface.TFrame")
        footer.grid(row=3, column=0, sticky="ew", padx=24, pady=(0, 14))
        footer.columnconfigure(1, weight=1)
        self.status_indicator = tk.Canvas(
            footer, width=12, height=12, bg=self.COLORS["surface"], highlightthickness=0
        )
        self.status_indicator.grid(row=0, column=0, padx=(12, 0))
        self.status_label = ttk.Label(footer, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.grid(row=0, column=1, sticky="w")
        ttk.Label(
            footer, text="Ctrl+R record   Ctrl+S stop   Ctrl+A answer",
            style="Status.TLabel"
        ).grid(row=0, column=2, sticky="e")
        self.work_progress = ttk.Progressbar(
            footer, mode="indeterminate", style="Work.Horizontal.TProgressbar", length=90
        )
        self.work_progress.grid(row=0, column=3, padx=(8, 12))
        self.stop_button.configure(state="disabled")

    def _create_editor_card(
        self, parent: ttk.PanedWindow, title: str, subtitle: str
    ) -> tuple[tk.Text, ttk.Frame, ttk.Frame]:
        card = ttk.Frame(parent, style="Surface.TFrame", padding=14)
        card.columnconfigure(0, weight=1)
        card.rowconfigure(3 if title == "Your prompt" else 2, weight=1)
        heading = ttk.Frame(card, style="Surface.TFrame")
        heading.grid(row=0, column=0, sticky="ew")
        ttk.Label(heading, text=title, style="Section.TLabel").pack(side="left")
        ttk.Label(card, text=subtitle, style="Muted.Surface.TLabel").grid(
            row=1, column=0, sticky="w", pady=(2, 8)
        )
        if title == "Your prompt":
            context = ttk.Frame(card, style="Surface.TFrame")
            context.grid(row=2, column=0, sticky="ew", pady=(0, 8))
            context.columnconfigure(1, weight=1)
            ttk.Label(context, text="Role", style="Muted.Surface.TLabel").grid(
                row=0, column=0, padx=(0, 8)
            )
            ttk.Entry(
                context, textvariable=self.interview_role_var, style="Context.TEntry"
            ).grid(
                row=0, column=1, sticky="ew"
            )
            ttk.Label(context, text="Domain", style="Muted.Surface.TLabel").grid(
                row=0, column=2, padx=(10, 8)
            )
            ttk.Combobox(
                context,
                textvariable=self.interview_domain_var,
                values=(
                    "General", "Management", "Design", "Technology", "Finance",
                    "Marketing", "Operations", "Sales", "Human Resources",
                    "Customer Support", "Other",
                ),
                state="readonly",
                style="Context.TCombobox",
                width=16,
            ).grid(row=0, column=3, padx=(0, 8))
            ttk.Button(
                context, text="Improve", command=self.improve_transcription,
                style="Secondary.TButton"
            ).grid(row=0, column=4)
        else:
            card.grid_rowconfigure(2, minsize=0)
        text_row = 3 if title == "Your prompt" else 2
        actions_row = 4 if title == "Your prompt" else 3
        text = tk.Text(
            card, wrap="word", undo=True, padx=12, pady=10, relief="flat", borderwidth=0,
            bg=self.COLORS["input"], fg=self.COLORS["text"], insertbackground=self.COLORS["accent"],
            selectbackground=self.COLORS["accent_dark"], font=("Segoe UI", 10)
        )
        text.grid(row=text_row, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(card, orient="vertical", command=text.yview)
        scrollbar.grid(row=text_row, column=1, sticky="ns")
        text.configure(yscrollcommand=scrollbar.set)
        actions = ttk.Frame(card, style="Surface.TFrame")
        actions.grid(row=actions_row, column=0, sticky="ew", pady=(10, 0))
        if title == "Your prompt":
            ttk.Button(actions, text="Suggest answer", command=self.suggest_answer,
                       style="Primary.TButton").pack(side="right")
        return text, card, actions

    def start_recording(self) -> None:
        try:
            if self.orchestrator.start_recording():
                import time
                self._recording_started = time.monotonic()
                self.start_button.configure(state="disabled")
                self.stop_button.configure(state="normal")
        except RuntimeError as exc:
            self.show_error("Unable to start recording", str(exc))

    def stop_recording(self) -> None:
        audio = self.orchestrator.stop_recording()
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.level_var.set(0.0)
        if audio:
            self.status_var.set("Transcribing your recording...")
            self.work_progress.start(12)
            threading.Thread(target=self._transcribe, args=(audio,), daemon=True).start()

    def _transcribe(self, audio: bytes) -> None:
        result = self.orchestrator.transcribe_audio(audio)
        self.root.after(0, self._apply_transcription, result.text, result.error)

    def _apply_transcription(self, text: str, error: str | None) -> None:
        self.work_progress.stop()
        if error:
            self.show_error("Transcription failed", error)
            return
        self.transcription.delete("1.0", "end")
        self.transcription.insert("1.0", text)
        self.status_var.set("Transcription ready - review it, then suggest an answer")

    def suggest_answer(self) -> None:
        prompt = self.transcription.get("1.0", "end").strip()
        if not prompt:
            self.show_error("No question", "Record or enter a question before requesting an answer.")
            return
        self._set_response("")
        self.status_var.set("Generating a local answer...")
        self.work_progress.start(12)
        threading.Thread(target=self._generate, args=(prompt,), daemon=True).start()

    def improve_transcription(self) -> None:
        transcription = self.transcription.get("1.0", "end").strip()
        if not transcription:
            self.show_error("No transcription", "Record or enter text before improving it.")
            return
        self.status_var.set("Checking the transcription against the interview context...")
        self.work_progress.start(12)
        threading.Thread(
            target=self._improve_transcription,
            args=(transcription, self.interview_role_var.get(), self.interview_domain_var.get()),
            daemon=True,
        ).start()

    def _improve_transcription(
        self, transcription: str, interview_role: str, interview_domain: str
    ) -> None:
        try:
            improved = self.orchestrator.improve_transcription(
                transcription, interview_role, interview_domain
            )
            self.root.after(0, self._apply_improved_transcription, improved)
        except Exception as exc:
            self.root.after(0, self.show_error, "Transcription improvement failed", str(exc))

    def _apply_improved_transcription(self, text: str) -> None:
        self.work_progress.stop()
        if not text.strip():
            self.show_error("No correction returned", "Ollama returned an empty correction.")
            return
        self.transcription.delete("1.0", "end")
        self.transcription.insert("1.0", text.strip())
        self.status_var.set("Transcription improved - review the suggested wording")

    def _generate(self, prompt: str) -> None:
        try:
            for token in self.orchestrator.generate_response(prompt):
                self.root.after(0, self.update_response, token, False)
            self.root.after(0, self.update_response, "", True)
        except Exception as exc:
            self.root.after(0, self.show_error, "Answer generation failed", str(exc))

    def update_audio_level(self, level: float) -> None:
        self.root.after(0, self.level_var.set, level)

    def update_response(self, token: str, complete: bool = False) -> None:
        self.response.configure(state="normal")
        self.response.insert("end", token)
        self.response.configure(state="disabled")
        if complete:
            self.work_progress.stop()
            self.status_var.set("Answer complete")

    def show_error(self, message: str, details: str = "") -> None:
        self.work_progress.stop()
        messagebox.showerror(message, details or message, parent=self.root)

    def show_warning(self, message: str) -> bool:
        return messagebox.askyesno("Confirm", message, parent=self.root)

    def _set_response(self, text: str) -> None:
        self.response.configure(state="normal")
        self.response.delete("1.0", "end")
        if text:
            self.response.insert("1.0", text)
        self.response.configure(state="disabled")

    def _copy_response(self) -> None:
        text = self.response.get("1.0", "end").strip()
        if not text:
            self.status_var.set("Nothing to copy")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set("Response copied to clipboard")

    def _clear_response(self) -> None:
        self._set_response("")
        self.status_var.set("Response cleared")

    def _poll_status(self) -> None:
        status = self.orchestrator.get_status()
        self.status_var.set(status.message)
        gpu = status.gpu
        self.gpu_var.set(f"GPU: {gpu.name}" if gpu.available else "GPU: CPU mode")
        self.connection_var.set(
            "Ollama: connected" if status.ollama_connected else "Ollama: offline"
        )
        color = {
            AppState.RECORDING: self.COLORS["danger"],
            AppState.ERROR: self.COLORS["danger"],
            AppState.TRANSCRIBING: self.COLORS["accent"],
            AppState.GENERATING: self.COLORS["accent"],
        }.get(status.state, self.COLORS["success"])
        self.status_indicator.delete("all")
        self.status_indicator.create_oval(2, 2, 10, 10, fill=color, outline=color)
        if status.state == AppState.RECORDING:
            import time
            elapsed = min(int(time.monotonic() - self._recording_started), self.orchestrator.config.audio.max_duration)
            self.timer_var.set(f"{elapsed // 60:02d}:{elapsed % 60:02d} / 01:00")
        else:
            self.timer_var.set("00:00 / 01:00")
        self._poll_id = self.root.after(self.orchestrator.config.ui.update_frequency, self._poll_status)

    def _close(self) -> None:
        if self.orchestrator.recorder.is_recording() and not self.show_warning(
            "A recording is active. Stop it and exit?"
        ):
            return
        self.orchestrator.shutdown(force=True)
        if self._poll_id:
            self.root.after_cancel(self._poll_id)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
