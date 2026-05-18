import sqlite3
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

class DetectionDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    abs_path TEXT UNIQUE,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    fps REAL,
                    duration REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER,
                    frame_index INTEGER,
                    t_seconds REAL,
                    species_name TEXT,
                    confidence REAL,
                    bbox_xyxy TEXT,
                    FOREIGN KEY(video_id) REFERENCES videos(id)
                )
            """)
            conn.commit()

    def save_video_results(self, video_data: Dict[str, Any]):
        abs_path = video_data["video_path"]
        filename = os.path.basename(abs_path)
        fps = video_data.get("video_fps", 0)
        duration = video_data.get("duration_seconds", 0)

        with self._get_conn() as conn:
            # Insert or update video
            cursor = conn.execute(
                "INSERT OR REPLACE INTO videos (filename, abs_path, fps, duration) VALUES (?, ?, ?, ?)",
                (filename, abs_path, fps, duration)
            )
            video_id = cursor.lastrowid

            # Clear old detections if updating
            conn.execute("DELETE FROM detections WHERE video_id = ?", (video_id,))

            # Insert detections
            detections_to_insert = []
            for frame in video_data.get("frames", []):
                frame_idx = frame["frame_index"]
                t_sec = frame["t_seconds"]
                for det in frame.get("detections", []):
                    species = det.get("speciesnet", {}).get("class_name", "Animal")
                    conf = det["confidence"]
                    bbox = ",".join(map(str, det["bbox_xyxy"]))
                    detections_to_insert.append((video_id, frame_idx, t_sec, species, conf, bbox))

            if detections_to_insert:
                conn.executemany(
                    "INSERT INTO detections (video_id, frame_index, t_seconds, species_name, confidence, bbox_xyxy) VALUES (?, ?, ?, ?, ?, ?)",
                    detections_to_insert
                )
            conn.commit()

    def get_all_species_counts(self) -> Dict[str, int]:
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT species_name, COUNT(*) FROM detections GROUP BY species_name")
            return {row[0]: row[1] for row in cursor.fetchall()}

    def get_video_stats(self, abs_path: str) -> Dict[str, Any]:
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT id FROM videos WHERE abs_path = ?", (abs_path,))
            row = cursor.fetchone()
            if not row:
                return {}
            video_id = row[0]

            # Get unique species count
            cursor = conn.execute("SELECT COUNT(DISTINCT species_name) FROM detections WHERE video_id = ?", (video_id,))
            unique_species = cursor.fetchone()[0]

            # Get top species
            cursor = conn.execute("SELECT DISTINCT species_name FROM detections WHERE video_id = ? LIMIT 3", (video_id,))
            top_species = [r[0] for r in cursor.fetchall()]

            return {
                "unique_species_count": unique_species,
                "top_species": top_species
            }

    def get_video_frames(self, abs_path: str) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT id FROM videos WHERE abs_path = ?", (abs_path,))
            row = cursor.fetchone()
            if not row:
                return []
            video_id = row[0]

            cursor = conn.execute("""
                SELECT frame_index, t_seconds, species_name, confidence, bbox_xyxy 
                FROM detections WHERE video_id = ? 
                ORDER BY frame_index ASC
            """, (video_id,))
            
            frames_dict = {}
            for row in cursor.fetchall():
                f_idx, t_sec, species, conf, bbox_str = row
                if f_idx not in frames_dict:
                    frames_dict[f_idx] = {
                        "frame_index": f_idx,
                        "t_seconds": t_sec,
                        "detections": []
                    }
                
                bbox = [float(x) for x in bbox_str.split(",")]
                det = {
                    "bbox_xyxy": bbox,
                    "confidence": conf,
                    "speciesnet": {"class_name": species, "probability": 1.0} # Probability not stored separately yet
                }
                frames_dict[f_idx]["detections"].append(det)
            
            return sorted(frames_dict.values(), key=lambda x: x["frame_index"])
