"""
logger.py — Ghi loss, accuracy, lr qua từng epoch ra file JSON/CSV
"""
import os
import json
import csv
from datetime import datetime


class TrainLogger:
    """
    Ghi log training ra 2 dạng:
    - <log_dir>/<run_name>/log.json  : toàn bộ history
    - <log_dir>/<run_name>/log.csv   : dạng bảng, dễ mở Excel/pandas
    """

    def __init__(self, log_dir: str, run_name: str = None):
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self.json_path = os.path.join(self.run_dir, "log.json")
        self.csv_path  = os.path.join(self.run_dir, "log.csv")
        self.history   = []
        self._csv_header_written = False

    def log(self, epoch: int, metrics: dict):
        """
        Ghi một epoch.

        Args:
            epoch:   số epoch (bắt đầu từ 1)
            metrics: dict, ví dụ:
                     {"train_loss": 0.5, "train_acc1": 80.0,
                      "val_loss": 0.4, "val_acc1": 82.0, "lr": 0.001}
        """
        entry = {"epoch": epoch, **metrics}
        self.history.append(entry)

        # Ghi JSON (overwrite toàn bộ để luôn có file hợp lệ)
        with open(self.json_path, "w") as f:
            json.dump(self.history, f, indent=2)

        # Ghi CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=entry.keys())
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(entry)

    def save_result(self, result_path: str, best_metrics: dict):
        """
        Lưu kết quả tốt nhất vào file result (dùng cho ablation).

        Args:
            result_path:  đường dẫn file json kết quả, vd 'results/step1_baseline.json'
            best_metrics: dict chứa best_acc, params, flops, ...
        """
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(best_metrics, f, indent=2)
        print(f"[Logger] Kết quả đã lưu vào: {result_path}")

    def print_epoch(self, epoch: int, total_epochs: int, metrics: dict):
        """In log ra terminal theo format gọn."""
        parts = [f"Epoch [{epoch}/{total_epochs}]"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}: {v:.4f}")
            else:
                parts.append(f"{k}: {v}")
        print("  ".join(parts))
