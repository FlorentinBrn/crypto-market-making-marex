from __future__ import annotations

import csv
import queue
import threading
from pathlib import Path
from typing import Iterable


class _FlushMarker:
    """Message de synchronisation envoyé par flush().

    Quand le thread writer rencontre ce marker dans la queue, il écrit
    immédiatement tout son buffer puis signale l'Event associé pour
    débloquer le caller de flush(). Permet un flush synchrone sans avoir
    à abuser de queue.join() (qui pose problème quand task_done n'est pas
    toujours appelé en symétrique).
    """

    __slots__ = ("event",)

    def __init__(self) -> None:
        self.event = threading.Event()


class CsvAppendWriter:
    """Writer CSV append-only, ASYNCHRONE pour ne pas bloquer le thread WS.

    Conception :
    - append() dépose une ligne dans une queue, retour immédiat ;
    - un thread writer dédié dépile par batches et écrit sur le disque,
      sortant l'I/O du chemin critique des messages WebSocket ;
    - flush() bloque jusqu'à ce que tout ce qui a été appendé avant ait
      effectivement été écrit (synchronisé via un _FlushMarker+Event) ;
    - close() demande l'arrêt propre du thread writer après flush final.
    """

    _SENTINEL = object()

    def __init__(
        self,
        output_path: Path,
        flush_size: int = 250,
        queue_max: int = 50_000,
    ) -> None:
        self.output_path = Path(output_path)
        self.flush_size = max(1, int(flush_size))
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self._queue: queue.Queue = queue.Queue(maxsize=queue_max)
        self._fieldnames: list[str] | None = None
        self._stopped = False

        self._writer_thread = threading.Thread(
            target=self._run, name=f"csv-{self.output_path.name}", daemon=True
        )
        self._writer_thread.start()

    # ------------------------------------------------------------------
    # API producteur
    # ------------------------------------------------------------------
    def append(self, row: dict) -> None:
        """Dépose une ligne dans la queue (non-bloquant dans le cas normal)."""
        if self._stopped:
            return
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self._queue.put(row, timeout=1.0)

    def extend(self, rows: Iterable[dict]) -> None:
        for row in rows:
            self.append(row)

    def flush(self, timeout: float = 10.0) -> None:
        """Attend que tout ce qui a été appendé avant soit écrit.

        Utilise un marker + Event pour éviter de dépendre de task_done().
        """
        if self._stopped:
            return
        marker = _FlushMarker()
        try:
            self._queue.put(marker, timeout=timeout)
        except queue.Full:
            return
        marker.event.wait(timeout=timeout)

    def close(self, timeout: float = 5.0) -> None:
        """Arrête proprement le thread writer."""
        if self._stopped:
            return
        self._stopped = True
        try:
            self._queue.put(self._SENTINEL, timeout=timeout)
        except queue.Full:
            pass
        self._writer_thread.join(timeout=timeout)

    @property
    def fieldnames(self) -> list[str] | None:
        return list(self._fieldnames) if self._fieldnames else None

    # ------------------------------------------------------------------
    # Thread writer
    # ------------------------------------------------------------------
    def _run(self) -> None:
        buffer: list[dict] = []
        f = None
        writer: csv.DictWriter | None = None

        def flush_buffer() -> None:
            """Écrit le buffer sur disque en ouvrant le fichier si besoin."""
            nonlocal f, writer
            if not buffer:
                return
            if f is None:
                # Premier écrit : union des colonnes vues → header.
                keys: list[str] = []
                seen: set[str] = set()
                for row in buffer:
                    for k in row.keys():
                        if k not in seen:
                            seen.add(k)
                            keys.append(k)
                self._fieldnames = keys
                f = self.output_path.open("w", newline="", encoding="utf-8")
                writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
            for row in buffer:
                writer.writerow(row)  # type: ignore[union-attr]
            f.flush()
            buffer.clear()

        try:
            while True:
                item = self._queue.get()
                if item is self._SENTINEL:
                    flush_buffer()
                    return
                if isinstance(item, _FlushMarker):
                    flush_buffer()
                    item.event.set()
                    continue
                buffer.append(item)
                if len(buffer) >= self.flush_size:
                    flush_buffer()
        finally:
            try:
                flush_buffer()
            finally:
                if f is not None:
                    f.close()


def write_dataframe_like(path: Path, rows: list[dict]) -> None:
    """Écrit une liste de dicts dans un CSV (synchrone — usage snapshots finaux)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
