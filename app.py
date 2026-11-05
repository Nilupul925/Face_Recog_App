import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

import cv2
from PIL import Image, ImageTk

from core.config import (
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_SAMPLES_PER_USER,
    EXPORT_DIR,
    SAMPLE_INTERVAL_MS,
    ensure_directories,
)
from core.database import DatabaseManager
from core.face_engine import (
    DetectedPersonResult,
    FaceRecognitionEngine,
    draw_enterprise_hud,
)


class EnterpriseFaceApp:
    """Modern Enterprise GUI for Simultaneous Multi-Person Face Recognition, Enrollment & Audit Logging."""

    # Modern Dark Slate Color Palette
    BG_PRIMARY = "#0f172a"      # Deep Slate
    BG_SECONDARY = "#1e293b"    # Card / Panel Slate
    BG_TERTIARY = "#334155"     # Input / Hover Slate
    FG_PRIMARY = "#f8fafc"      # Crisp White
    FG_SECONDARY = "#94a3b8"    # Muted Slate Text
    ACCENT_BLUE = "#3b82f6"     # Primary Action Blue
    ACCENT_GREEN = "#10b981"    # Verified Emerald
    ACCENT_AMBER = "#f59e0b"    # Warning / Enrollment Amber
    ACCENT_RED = "#ef4444"      # Danger / Unknown Crimson

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("VisionGuard Enterprise — Multi-Person Face Recognition Suite")
        self.root.geometry("1360x820")
        self.root.minsize(1160, 720)
        self.root.configure(bg=self.BG_PRIMARY)

        ensure_directories()
        self.db = DatabaseManager()
        self.engine = FaceRecognitionEngine(self.db)

        # Camera & Stream State
        self.camera_index = tk.IntVar(value=0)
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_running = False
        self.current_frame_bgr = None
        self.fps = 0.0
        self._last_frame_ts = time.perf_counter()

        # Enrollment State
        self.enrolling = False
        self.enroll_person_id: Optional[int] = None
        self.enroll_person_name: str = ""
        self.enroll_count = 0
        self.enroll_target = DEFAULT_SAMPLES_PER_USER
        self.last_sample_time_ms = 0.0
        self.auto_train_var = tk.BooleanVar(value=True)

        # Engine Settings Variables
        self.threshold_var = tk.DoubleVar(value=DEFAULT_DISTANCE_THRESHOLD)
        self.equalize_var = tk.BooleanVar(value=True)
        self.search_var = tk.StringVar()

        self._configure_styles()
        self._build_layout()
        self.refresh_user_table()
        self.refresh_log_table()
        self._update_status_badges()

        # Automatically start camera on launch
        self.root.after(250, self.start_camera)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(".", background=self.BG_PRIMARY, foreground=self.FG_PRIMARY, font=("Segoe UI", 10))
        style.configure("TFrame", background=self.BG_PRIMARY)
        style.configure("Card.TFrame", background=self.BG_SECONDARY)
        style.configure("TLabel", background=self.BG_SECONDARY, foreground=self.FG_PRIMARY, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=self.BG_SECONDARY, foreground=self.FG_PRIMARY, font=("Segoe UI Semibold", 15))
        style.configure("SubHeader.TLabel", background=self.BG_SECONDARY, foreground=self.FG_SECONDARY, font=("Segoe UI", 9))
        style.configure("Section.TLabel", background=self.BG_SECONDARY, foreground=self.ACCENT_BLUE, font=("Segoe UI Semibold", 11))

        # Notebook tabs
        style.configure("TNotebook", background=self.BG_SECONDARY, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=self.BG_PRIMARY,
            foreground=self.FG_SECONDARY,
            padding=(14, 8),
            font=("Segoe UI Semibold", 10),
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.BG_SECONDARY), ("active", self.BG_TERTIARY)],
            foreground=[("selected", self.FG_PRIMARY), ("active", self.FG_PRIMARY)],
        )

        # Treeview tables
        style.configure(
            "Treeview",
            background=self.BG_PRIMARY,
            foreground=self.FG_PRIMARY,
            fieldbackground=self.BG_PRIMARY,
            rowheight=28,
            borderwidth=0,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Treeview.Heading",
            background=self.BG_TERTIARY,
            foreground=self.FG_PRIMARY,
            font=("Segoe UI Semibold", 9),
            relief="flat",
            padding=(6, 5),
        )
        style.map("Treeview", background=[("selected", self.ACCENT_BLUE)])

        # Progressbar
        style.configure(
            "Amber.Horizontal.TProgressbar",
            troughcolor=self.BG_PRIMARY,
            background=self.ACCENT_AMBER,
            thickness=14,
        )

    def _build_layout(self) -> None:
        # 1. Top Enterprise Header Bar
        header = ttk.Frame(self.root, style="Card.TFrame", padding=(18, 12))
        header.pack(fill=tk.X, padx=14, pady=(12, 6))

        title_box = ttk.Frame(header, style="Card.TFrame")
        title_box.pack(side=tk.LEFT)
        ttk.Label(
            title_box,
            text="VisionGuard Enterprise",
            style="Header.TLabel",
        ).pack(anchor=tk.W)
        ttk.Label(
            title_box,
            text="Simultaneous Multi-Person Face Detection, LBPH Recognition & Attendance Suite",
            style="SubHeader.TLabel",
        ).pack(anchor=tk.W)

        # Quick Actions & Status Badges on Right of Header
        actions_box = ttk.Frame(header, style="Card.TFrame")
        actions_box.pack(side=tk.RIGHT)

        self.model_badge = tk.Label(
            actions_box,
            text="MODEL: CHECKING",
            bg=self.BG_TERTIARY,
            fg=self.FG_PRIMARY,
            font=("Segoe UI Semibold", 9),
            padx=10,
            pady=5,
        )
        self.model_badge.pack(side=tk.LEFT, padx=6)

        self.users_badge = tk.Label(
            actions_box,
            text="USERS: 0",
            bg=self.BG_TERTIARY,
            fg=self.FG_PRIMARY,
            font=("Segoe UI Semibold", 9),
            padx=10,
            pady=5,
        )
        self.users_badge.pack(side=tk.LEFT, padx=6)

        self.btn_cam_toggle = tk.Button(
            actions_box,
            text="Start Camera",
            command=self.toggle_camera,
            bg=self.ACCENT_GREEN,
            fg=self.FG_PRIMARY,
            activebackground="#059669",
            activeforeground=self.FG_PRIMARY,
            font=("Segoe UI Semibold", 9),
            relief=tk.FLAT,
            padx=12,
            pady=5,
            cursor="hand2",
        )
        self.btn_cam_toggle.pack(side=tk.LEFT, padx=6)

        self.btn_train_top = tk.Button(
            actions_box,
            text="Train Model Now",
            command=self.run_training,
            bg=self.ACCENT_BLUE,
            fg=self.FG_PRIMARY,
            activebackground="#2563eb",
            activeforeground=self.FG_PRIMARY,
            font=("Segoe UI Semibold", 9),
            relief=tk.FLAT,
            padx=12,
            pady=5,
            cursor="hand2",
        )
        self.btn_train_top.pack(side=tk.LEFT, padx=6)

        self.btn_snapshot = tk.Button(
            actions_box,
            text="Save Snapshot",
            command=self.save_snapshot,
            bg=self.BG_TERTIARY,
            fg=self.FG_PRIMARY,
            activebackground="#475569",
            activeforeground=self.FG_PRIMARY,
            font=("Segoe UI Semibold", 9),
            relief=tk.FLAT,
            padx=12,
            pady=5,
            cursor="hand2",
        )
        self.btn_snapshot.pack(side=tk.LEFT, padx=(6, 0))

        # 2. Main Split Content Area
        main_body = ttk.Frame(self.root)
        main_body.pack(fill=tk.BOTH, expand=True, padx=14, pady=6)

        # Left Column: Live Camera Viewport + Simultaneous Multi-Person Strip
        left_panel = ttk.Frame(main_body, style="Card.TFrame", padding=12)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.video_label = tk.Label(
            left_panel,
            bg="#090d16",
            fg=self.FG_SECONDARY,
            text="Initializing Camera Stream...",
            font=("Segoe UI Semibold", 13),
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Live Simultaneous Multi-Person Status Bar under Video
        multi_bar = ttk.Frame(left_panel, style="Card.TFrame", padding=(4, 8, 4, 2))
        multi_bar.pack(fill=tk.X, pady=(8, 0))

        ttk.Label(
            multi_bar,
            text="Simultaneous Live Detections in Frame:",
            style="Section.TLabel",
        ).pack(anchor=tk.W)

        self.live_faces_var = tk.StringVar(
            value="No faces currently in frame. Stand in front of the camera to detect one or multiple people simultaneously."
        )
        self.live_faces_label = tk.Label(
            multi_bar,
            textvariable=self.live_faces_var,
            bg=self.BG_PRIMARY,
            fg=self.FG_PRIMARY,
            font=("Consolas", 10),
            anchor=tk.W,
            justify=tk.LEFT,
            padx=10,
            pady=8,
        )
        self.live_faces_label.pack(fill=tk.X, pady=(4, 0))

        # Right Column: Enterprise Control Tabs
        right_panel = ttk.Frame(main_body, style="Card.TFrame", width=460)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)

        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.tab_enroll = ttk.Frame(self.notebook, style="Card.TFrame", padding=14)
        self.tab_users = ttk.Frame(self.notebook, style="Card.TFrame", padding=14)
        self.tab_logs = ttk.Frame(self.notebook, style="Card.TFrame", padding=14)
        self.tab_settings = ttk.Frame(self.notebook, style="Card.TFrame", padding=14)

        self.notebook.add(self.tab_enroll, text="Enroll User")
        self.notebook.add(self.tab_users, text="Users & Dataset")
        self.notebook.add(self.tab_logs, text="Audit Logs")
        self.notebook.add(self.tab_settings, text="Settings")

        self._build_enroll_tab()
        self._build_users_tab()
        self._build_logs_tab()
        self._build_settings_tab()

        # 3. Bottom Status Bar
        self.status_var = tk.StringVar(value="System ready.")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            bg=self.BG_SECONDARY,
            fg=self.FG_SECONDARY,
            anchor=tk.W,
            font=("Segoe UI", 9),
            padx=16,
            pady=5,
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _create_entry(self, parent: ttk.Frame, label_text: str, default_val: str = "") -> tk.Entry:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text=label_text, width=16, anchor=tk.W).pack(side=tk.LEFT)
        entry = tk.Entry(
            row,
            bg=self.BG_PRIMARY,
            fg=self.FG_PRIMARY,
            insertbackground=self.FG_PRIMARY,
            relief=tk.FLAT,
            font=("Segoe UI", 10),
            highlightthickness=1,
            highlightbackground=self.BG_TERTIARY,
            highlightcolor=self.ACCENT_BLUE,
        )
        entry.pack(side=tk.RIGHT, fill=tk.X, expand=True, ipady=5, padx=(6, 0))
        if default_val:
            entry.insert(0, default_val)
        return entry

    def _build_enroll_tab(self) -> None:
        ttk.Label(
            self.tab_enroll,
            text="Register / Update Person Profile",
            style="Section.TLabel",
        ).pack(anchor=tk.W, pady=(0, 8))

        self.entry_id = self._create_entry(self.tab_enroll, "User ID (Numeric):")
        self.entry_name = self._create_entry(self.tab_enroll, "Full Name:")
        self.entry_age = self._create_entry(self.tab_enroll, "Age (Years):")
        self.entry_dept = self._create_entry(self.tab_enroll, "Department:", "Engineering")
        self.entry_role = self._create_entry(self.tab_enroll, "Role / Title:", "Staff")
        self.entry_samples = self._create_entry(
            self.tab_enroll, "Target Samples:", str(DEFAULT_SAMPLES_PER_USER)
        )

        chk = tk.Checkbutton(
            self.tab_enroll,
            text="Automatically train model after sample capture",
            variable=self.auto_train_var,
            bg=self.BG_SECONDARY,
            fg=self.FG_PRIMARY,
            selectcolor=self.BG_PRIMARY,
            activebackground=self.BG_SECONDARY,
            activeforeground=self.FG_PRIMARY,
            font=("Segoe UI", 9),
        )
        chk.pack(anchor=tk.W, pady=(8, 6))

        btn_row = ttk.Frame(self.tab_enroll, style="Card.TFrame")
        btn_row.pack(fill=tk.X, pady=8)

        self.btn_start_enroll = tk.Button(
            btn_row,
            text="Start Live Face Capture",
            command=self.start_enrollment,
            bg=self.ACCENT_AMBER,
            fg="#0f172a",
            activebackground="#d97706",
            font=("Segoe UI Semibold", 10),
            relief=tk.FLAT,
            pady=7,
            cursor="hand2",
        )
        self.btn_start_enroll.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_save_profile_only = tk.Button(
            btn_row,
            text="Save Profile Only",
            command=self.save_profile_only,
            bg=self.BG_TERTIARY,
            fg=self.FG_PRIMARY,
            activebackground="#475569",
            font=("Segoe UI Semibold", 10),
            relief=tk.FLAT,
            pady=7,
            cursor="hand2",
        )
        self.btn_save_profile_only.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        # Progress section
        ttk.Label(
            self.tab_enroll,
            text="Enrollment Progress:",
            style="SubHeader.TLabel",
        ).pack(anchor=tk.W, pady=(12, 4))

        self.enroll_progress_bar = ttk.Progressbar(
            self.tab_enroll,
            style="Amber.Horizontal.TProgressbar",
            orient=tk.HORIZONTAL,
            mode="determinate",
        )
        self.enroll_progress_bar.pack(fill=tk.X)

        self.enroll_status_label = ttk.Label(
            self.tab_enroll,
            text="Idle — Fill user details above and click 'Start Live Face Capture'.",
            style="SubHeader.TLabel",
            wraplength=400,
        )
        self.enroll_status_label.pack(anchor=tk.W, pady=(6, 0))

        # Helpful guidance card
        tips = (
            "Enterprise Enrollment Tips:\n"
            "• Ensure good frontal lighting on the subject's face.\n"
            "• Gently turn head slightly left, right, and smile during capture.\n"
            "• Only the closest/largest face is captured to prevent bystander contamination."
        )
        tip_lbl = tk.Label(
            self.tab_enroll,
            text=tips,
            bg=self.BG_PRIMARY,
            fg=self.FG_SECONDARY,
            justify=tk.LEFT,
            anchor=tk.W,
            font=("Segoe UI", 9),
            padx=10,
            pady=10,
        )
        tip_lbl.pack(fill=tk.X, pady=(18, 0))

    def _build_users_tab(self) -> None:
        top_row = ttk.Frame(self.tab_users, style="Card.TFrame")
        top_row.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(top_row, text="Search:", width=8).pack(side=tk.LEFT)
        search_entry = tk.Entry(
            top_row,
            textvariable=self.search_var,
            bg=self.BG_PRIMARY,
            fg=self.FG_PRIMARY,
            insertbackground=self.FG_PRIMARY,
            relief=tk.FLAT,
            font=("Segoe UI", 9),
        )
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=4)
        self.search_var.trace_add("write", lambda *_: self.refresh_user_table())

        cols = ("id", "name", "age", "dept", "samples")
        self.user_tree = ttk.Treeview(self.tab_users, columns=cols, show="headings", height=14)
        self.user_tree.heading("id", text="ID")
        self.user_tree.heading("name", text="Name")
        self.user_tree.heading("age", text="Age")
        self.user_tree.heading("dept", text="Department")
        self.user_tree.heading("samples", text="Samples")

        self.user_tree.column("id", width=45, anchor=tk.CENTER)
        self.user_tree.column("name", width=125, anchor=tk.W)
        self.user_tree.column("age", width=45, anchor=tk.CENTER)
        self.user_tree.column("dept", width=105, anchor=tk.W)
        self.user_tree.column("samples", width=65, anchor=tk.CENTER)
        self.user_tree.pack(fill=tk.BOTH, expand=True, pady=4)

        btn_bar = ttk.Frame(self.tab_users, style="Card.TFrame")
        btn_bar.pack(fill=tk.X, pady=(8, 0))

        tk.Button(
            btn_bar,
            text="Edit Selected",
            command=self.load_selected_user_to_form,
            bg=self.ACCENT_BLUE,
            fg=self.FG_PRIMARY,
            relief=tk.FLAT,
            font=("Segoe UI Semibold", 9),
            padx=10,
            pady=5,
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            btn_bar,
            text="Delete Selected",
            command=self.delete_selected_user,
            bg=self.ACCENT_RED,
            fg=self.FG_PRIMARY,
            relief=tk.FLAT,
            font=("Segoe UI Semibold", 9),
            padx=10,
            pady=5,
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            btn_bar,
            text="Refresh",
            command=self.refresh_user_table,
            bg=self.BG_TERTIARY,
            fg=self.FG_PRIMARY,
            relief=tk.FLAT,
            font=("Segoe UI Semibold", 9),
            padx=10,
            pady=5,
            cursor="hand2",
        ).pack(side=tk.RIGHT)

    def _build_logs_tab(self) -> None:
        ttk.Label(
            self.tab_logs,
            text="Real-Time Recognition & Attendance Audit Log",
            style="Section.TLabel",
        ).pack(anchor=tk.W, pady=(0, 8))

        cols = ("time", "id", "name", "dept", "match")
        self.log_tree = ttk.Treeview(self.tab_logs, columns=cols, show="headings", height=15)
        self.log_tree.heading("time", text="Timestamp")
        self.log_tree.heading("id", text="ID")
        self.log_tree.heading("name", text="Name")
        self.log_tree.heading("dept", text="Department")
        self.log_tree.heading("match", text="Match")

        self.log_tree.column("time", width=130, anchor=tk.CENTER)
        self.log_tree.column("id", width=40, anchor=tk.CENTER)
        self.log_tree.column("name", width=110, anchor=tk.W)
        self.log_tree.column("dept", width=85, anchor=tk.W)
        self.log_tree.column("match", width=60, anchor=tk.CENTER)
        self.log_tree.pack(fill=tk.BOTH, expand=True, pady=4)

        btn_bar = ttk.Frame(self.tab_logs, style="Card.TFrame")
        btn_bar.pack(fill=tk.X, pady=(8, 0))

        tk.Button(
            btn_bar,
            text="Export CSV",
            command=self.export_logs_csv,
            bg=self.ACCENT_GREEN,
            fg=self.FG_PRIMARY,
            relief=tk.FLAT,
            font=("Segoe UI Semibold", 9),
            padx=10,
            pady=5,
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            btn_bar,
            text="Clear Logs",
            command=self.clear_logs,
            bg=self.ACCENT_RED,
            fg=self.FG_PRIMARY,
            relief=tk.FLAT,
            font=("Segoe UI Semibold", 9),
            padx=10,
            pady=5,
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            btn_bar,
            text="Refresh",
            command=self.refresh_log_table,
            bg=self.BG_TERTIARY,
            fg=self.FG_PRIMARY,
            relief=tk.FLAT,
            font=("Segoe UI Semibold", 9),
            padx=10,
            pady=5,
            cursor="hand2",
        ).pack(side=tk.RIGHT)

    def _build_settings_tab(self) -> None:
        ttk.Label(
            self.tab_settings,
            text="Recognition & Camera Configuration",
            style="Section.TLabel",
        ).pack(anchor=tk.W, pady=(0, 12))

        # Camera selector
        cam_row = ttk.Frame(self.tab_settings, style="Card.TFrame")
        cam_row.pack(fill=tk.X, pady=6)
        ttk.Label(cam_row, text="Camera Device Index:", width=22).pack(side=tk.LEFT)
        cam_spin = tk.Spinbox(
            cam_row,
            from_=0,
            to=5,
            textvariable=self.camera_index,
            width=6,
            bg=self.BG_PRIMARY,
            fg=self.FG_PRIMARY,
            relief=tk.FLAT,
            font=("Segoe UI", 10),
        )
        cam_spin.pack(side=tk.LEFT, padx=6)
        tk.Button(
            cam_row,
            text="Reconnect Camera",
            command=self.restart_camera,
            bg=self.ACCENT_BLUE,
            fg=self.FG_PRIMARY,
            relief=tk.FLAT,
            font=("Segoe UI Semibold", 9),
            padx=10,
            pady=4,
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=6)

        # LBPH Distance Threshold slider
        thresh_frame = ttk.Frame(self.tab_settings, style="Card.TFrame")
        thresh_frame.pack(fill=tk.X, pady=14)

        self.thresh_label = ttk.Label(
            thresh_frame,
            text=f"LBPH Distance Threshold: {self.threshold_var.get():.1f} (Recommended: 60 - 75)",
        )
        self.thresh_label.pack(anchor=tk.W)

        scale = tk.Scale(
            thresh_frame,
            from_=35.0,
            to=105.0,
            resolution=1.0,
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            command=self._on_threshold_change,
            bg=self.BG_SECONDARY,
            fg=self.FG_PRIMARY,
            highlightthickness=0,
            troughcolor=self.BG_PRIMARY,
        )
        scale.pack(fill=tk.X, pady=4)

        ttk.Label(
            thresh_frame,
            text="Note: In OpenCV LBPH, lower distance = stricter matching. Raising above 85 may cause false positives.",
            style="SubHeader.TLabel",
            wraplength=400,
        ).pack(anchor=tk.W)

        # Equalization checkbox
        eq_chk = tk.Checkbutton(
            self.tab_settings,
            text="Apply Histogram Equalization (Improves lighting robustness)",
            variable=self.equalize_var,
            command=self._on_equalize_change,
            bg=self.BG_SECONDARY,
            fg=self.FG_PRIMARY,
            selectcolor=self.BG_PRIMARY,
            activebackground=self.BG_SECONDARY,
            activeforeground=self.FG_PRIMARY,
            font=("Segoe UI", 9),
        )
        eq_chk.pack(anchor=tk.W, pady=10)

    def _on_threshold_change(self, _val: str) -> None:
        val = float(self.threshold_var.get())
        self.engine.distance_threshold = val
        self.thresh_label.configure(
            text=f"LBPH Distance Threshold: {val:.1f} (Recommended: 60 - 75)"
        )

    def _on_equalize_change(self) -> None:
        self.engine.apply_equalization = bool(self.equalize_var.get())

    def _update_status_badges(self) -> None:
        if self.engine.model_loaded:
            self.model_badge.configure(text="MODEL: READY", bg=self.ACCENT_GREEN, fg="#0f172a")
        else:
            self.model_badge.configure(text="MODEL: UNTRAINED", bg=self.ACCENT_AMBER, fg="#0f172a")

        total_users = len(self.db.list_people())
        self.users_badge.configure(text=f"USERS: {total_users}")

    # ---------------- Camera & Real-Time Loop ----------------

    def start_camera(self) -> None:
        if self.camera_running:
            return
        idx = int(self.camera_index.get())
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(idx)

        if not self.cap.isOpened():
            self.status_var.set(f"Error: Could not open camera at index {idx}.")
            self.video_label.configure(
                image="",
                text=f"Camera (Index {idx}) Unavailable.\nCheck webcam connection or change Camera Index in Settings.",
            )
            return

        self.camera_running = True
        self.btn_cam_toggle.configure(text="Stop Camera", bg=self.ACCENT_RED)
        self.status_var.set(f"Camera {idx} active — Simultaneous multi-person recognition running.")
        self._last_frame_ts = time.perf_counter()
        self._video_loop()

    def stop_camera(self) -> None:
        self.camera_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.btn_cam_toggle.configure(text="Start Camera", bg=self.ACCENT_GREEN)
        self.video_label.configure(image="", text="Camera Stream Paused.")
        self.status_var.set("Camera stopped.")

    def toggle_camera(self) -> None:
        if self.camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def restart_camera(self) -> None:
        self.stop_camera()
        self.root.after(150, self.start_camera)

    def _video_loop(self) -> None:
        if not self.camera_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.status_var.set("Warning: Dropped frame from camera.")
            self.root.after(30, self._video_loop)
            return

        now = time.perf_counter()
        dt = max(now - self._last_frame_ts, 1e-4)
        self._last_frame_ts = now
        instant_fps = 1.0 / dt
        self.fps = (0.85 * self.fps + 0.15 * instant_fps) if self.fps > 0 else instant_fps

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Simultaneous multi-person detection & recognition
        detections: List[DetectedPersonResult] = self.engine.recognize_faces_in_frame(
            gray, record_attendance=(not self.enrolling)
        )

        # Handle active live enrollment sample capture
        enrollment_overlay = None
        if self.enrolling and self.enroll_person_id is not None:
            now_ms = time.perf_counter() * 1000.0
            if len(detections) > 0 and (now_ms - self.last_sample_time_ms >= SAMPLE_INTERVAL_MS):
                # Capture the largest/closest face (detections are sorted largest-first)
                primary_bbox = detections[0].bbox
                try:
                    self.engine.save_enrollment_sample(
                        self.enroll_person_id,
                        self.enroll_count + 1,
                        gray,
                        primary_bbox,
                    )
                    self.enroll_count += 1
                    self.last_sample_time_ms = now_ms
                    self.enroll_progress_bar["value"] = (
                        self.enroll_count / max(1, self.enroll_target)
                    ) * 100.0
                    multi_note = (
                        " (Multiple faces visible — isolating closest face)"
                        if len(detections) > 1
                        else ""
                    )
                    self.enroll_status_label.configure(
                        text=f"Captured sample {self.enroll_count}/{self.enroll_target} for {self.enroll_person_name}{multi_note}"
                    )
                except Exception as exc:
                    self.status_var.set(f"Error saving sample: {exc}")

            enrollment_overlay = (self.enroll_count, self.enroll_target, self.enroll_person_name)
            if self.enroll_count >= self.enroll_target:
                self._finish_enrollment()

        # Render HUD overlay
        annotated = draw_enterprise_hud(
            frame,
            detections,
            fps=self.fps,
            model_loaded=self.engine.model_loaded,
            enrollment_progress=enrollment_overlay,
        )
        self.current_frame_bgr = annotated

        # Update simultaneous multi-person summary strip
        self._update_simultaneous_summary(detections)

        # Fit frame cleanly inside the Tkinter label
        lbl_w = max(self.video_label.winfo_width(), 640)
        lbl_h = max(self.video_label.winfo_height(), 460)
        fh, fw = annotated.shape[:2]
        scale = min(lbl_w / fw, lbl_h / fh)
        disp_w, disp_h = max(1, int(fw * scale)), max(1, int(fh * scale))

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((disp_w, disp_h), Image.Resampling.BILINEAR)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        self.video_label.configure(image=tk_img, text="")
        self.video_label.image = tk_img

        self.root.after(15, self._video_loop)

    def _update_simultaneous_summary(self, detections: List[DetectedPersonResult]) -> None:
        if not detections:
            self.live_faces_var.set(
                "No faces currently in frame. Waiting for one or multiple people..."
            )
            return

        badges: List[str] = []
        for idx, det in enumerate(detections, start=1):
            if det.is_known and det.profile is not None:
                p = det.profile
                badges.append(
                    f"[{idx}] VERIFIED: {p.name} (ID:{p.id}, Age:{p.age}, {p.department}) — {det.similarity_pct:.0f}%"
                )
            else:
                badges.append(f"[{idx}] UNKNOWN SUBJECT (Dist: {det.distance:.1f})")

        self.live_faces_var.set("   |   ".join(badges))

    # ---------------- Enrollment & Training Actions ----------------

    def _parse_user_form(self):
        try:
            pid = int(self.entry_id.get().strip())
            if pid <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "User ID must be a positive integer.")
            return None

        name = self.entry_name.get().strip()
        if not name:
            messagebox.showerror("Invalid Input", "Please enter the person's Full Name.")
            return None

        try:
            age = int(self.entry_age.get().strip())
            if age <= 0 or age > 130:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Age must be a valid integer (1 - 130).")
            return None

        dept = self.entry_dept.get().strip() or "General"
        role = self.entry_role.get().strip() or "Member"

        try:
            target_samples = int(self.entry_samples.get().strip())
            if target_samples < 5 or target_samples > 200:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Target Samples must be between 5 and 200.")
            return None

        return pid, name, age, dept, role, target_samples

    def save_profile_only(self) -> None:
        parsed = self._parse_user_form()
        if not parsed:
            return
        pid, name, age, dept, role, _ = parsed
        self.db.upsert_person(pid, name, age, dept, role)
        self.refresh_user_table()
        self._update_status_badges()
        self.status_var.set(f"Saved profile for {name} (ID: {pid}).")
        messagebox.showinfo("Profile Saved", f"Profile for '{name}' (ID: {pid}) has been saved.")

    def start_enrollment(self) -> None:
        if self.enrolling:
            self.enrolling = False
            self.btn_start_enroll.configure(text="Start Live Face Capture", bg=self.ACCENT_AMBER)
            self.enroll_status_label.configure(text="Enrollment cancelled by user.")
            return

        parsed = self._parse_user_form()
        if not parsed:
            return
        pid, name, age, dept, role, target_samples = parsed

        if not self.camera_running:
            self.start_camera()
            if not self.camera_running:
                messagebox.showerror("Camera Error", "Cannot start enrollment without an active camera.")
                return

        # Upsert user profile in SQLite
        self.db.upsert_person(pid, name, age, dept, role)
        self.refresh_user_table()
        self._update_status_badges()

        self.enroll_person_id = pid
        self.enroll_person_name = name
        self.enroll_count = 0
        self.enroll_target = target_samples
        self.enroll_progress_bar["value"] = 0
        self.enrolling = True
        self.btn_start_enroll.configure(text="Stop Capture", bg=self.ACCENT_RED)
        self.enroll_status_label.configure(
            text=f"Capturing {target_samples} face samples for {name} (ID: {pid})... Look at the camera."
        )
        self.status_var.set(f"Enrolling {name} (ID: {pid})...")

    def _finish_enrollment(self) -> None:
        self.enrolling = False
        self.btn_start_enroll.configure(text="Start Live Face Capture", bg=self.ACCENT_AMBER)
        self.enroll_status_label.configure(
            text=f"Completed! Captured {self.enroll_count} samples for {self.enroll_person_name}."
        )
        self.refresh_user_table()

        if self.auto_train_var.get():
            self.run_training(silent_confirm=True)
        else:
            messagebox.showinfo(
                "Enrollment Complete",
                f"Captured {self.enroll_count} face samples for {self.enroll_person_name}.\n"
                "Click 'Train Model Now' to update the recognizer.",
            )

    def run_training(self, silent_confirm: bool = False) -> None:
        try:
            self.status_var.set("Training LBPH Face Recognizer on dataset...")
            self.root.update_idletasks()
            num_images, num_users = self.engine.train_model()
            self._update_status_badges()
            self.refresh_user_table()
            msg = f"Model trained on {num_images} samples across {num_users} user(s)."
            self.status_var.set(msg)
            if not silent_confirm:
                messagebox.showinfo("Training Complete", msg)
            else:
                self.enroll_status_label.configure(
                    text=f"Enrollment & Model Training Complete! ({num_images} total samples, {num_users} users)"
                )
        except Exception as exc:
            self.status_var.set(f"Training failed: {exc}")
            messagebox.showerror("Training Error", str(exc))

    # ---------------- Directory & Logs Management ----------------

    def refresh_user_table(self) -> None:
        for item in self.user_tree.get_children():
            self.user_tree.delete(item)

        query = self.search_var.get().strip().lower()
        sample_counts = self.engine.count_user_samples()

        for p in self.db.list_people():
            if query and (
                query not in str(p.id).lower()
                and query not in p.name.lower()
                and query not in p.department.lower()
            ):
                continue
            samples = sample_counts.get(p.id, 0)
            self.user_tree.insert(
                "",
                tk.END,
                values=(p.id, p.name, p.age, p.department, samples),
            )

    def load_selected_user_to_form(self) -> None:
        selected = self.user_tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a user from the table first.")
            return
        vals = self.user_tree.item(selected[0], "values")
        pid = int(vals[0])
        profile = self.db.get_person(pid)
        if not profile:
            return

        self.entry_id.delete(0, tk.END)
        self.entry_id.insert(0, str(profile.id))
        self.entry_name.delete(0, tk.END)
        self.entry_name.insert(0, profile.name)
        self.entry_age.delete(0, tk.END)
        self.entry_age.insert(0, str(profile.age))
        self.entry_dept.delete(0, tk.END)
        self.entry_dept.insert(0, profile.department)
        self.entry_role.delete(0, tk.END)
        self.entry_role.insert(0, profile.role)

        self.notebook.select(self.tab_enroll)
        self.status_var.set(f"Loaded user {profile.name} (ID: {profile.id}) into Enrollment form.")

    def delete_selected_user(self) -> None:
        selected = self.user_tree.selection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a user to delete.")
            return
        vals = self.user_tree.item(selected[0], "values")
        pid, name = int(vals[0]), vals[1]

        if not messagebox.askyesno(
            "Confirm Deletion",
            f"Delete user '{name}' (ID: {pid}) and all their face samples from the dataset?",
        ):
            return

        self.db.delete_person(pid)
        removed_imgs = self.engine.delete_user_samples(pid)
        self.refresh_user_table()
        self._update_status_badges()

        # Re-train model if other samples remain
        remaining = sum(self.engine.count_user_samples().values())
        if remaining > 0:
            try:
                self.engine.train_model()
            except Exception:
                pass
        self._update_status_badges()
        self.status_var.set(f"Deleted user {name} (ID: {pid}) and {removed_imgs} sample images.")

    def refresh_log_table(self) -> None:
        for item in self.log_tree.get_children():
            self.log_tree.delete(item)

        for entry in self.db.get_recent_logs(limit=150):
            self.log_tree.insert(
                "",
                tk.END,
                values=(
                    entry.timestamp,
                    entry.person_id,
                    entry.name,
                    entry.department,
                    f"{entry.similarity_pct:.0f}%",
                ),
            )

    def export_logs_csv(self) -> None:
        default_name = f"recognition_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = filedialog.asksaveasfilename(
            initialdir=str(EXPORT_DIR),
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title="Export Recognition Audit Logs",
        )
        if not file_path:
            return
        from pathlib import Path

        out = self.db.export_logs_to_csv(Path(file_path))
        self.status_var.set(f"Exported audit logs to {out}")
        messagebox.showinfo("Export Complete", f"Logs exported successfully to:\n{out}")

    def clear_logs(self) -> None:
        if messagebox.askyesno("Clear Audit Logs", "Are you sure you want to clear all recognition logs?"):
            self.db.clear_logs()
            self.refresh_log_table()
            self.status_var.set("Cleared all recognition logs.")

    def save_snapshot(self) -> None:
        if self.current_frame_bgr is None:
            messagebox.showwarning("Snapshot", "No active camera frame to capture.")
            return
        ensure_directories()
        snap_path = EXPORT_DIR / f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(snap_path), self.current_frame_bgr)
        self.status_var.set(f"Snapshot saved to {snap_path}")
        messagebox.showinfo("Snapshot Saved", f"Saved annotated frame to:\n{snap_path}")

    def on_close(self) -> None:
        self.stop_camera()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    EnterpriseFaceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

