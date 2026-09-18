"""Edge node entrypoint.

Usage::

    python -m veritrack_edge.run_node --config config/cam_thane_07.json
    python -m veritrack_edge.run_node --config cam.json --sink kafka \\
        --brokers 10.12.0.4:9092 --topic veritrack.passes.raw

The Kafka sink is optional at import time so the module runs on a bench with no
broker reachable. Delivery is at-least-once with the deterministic ``pass_id``
as the message key, which lets the Stage 2 gateway deduplicate on primary key
after a backhaul outage.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Protocol

from .config import PipelineConfig, load_config
from .errors import FatalEdgeError, PayloadTooLargeError, StreamError
from .ingest import VideoStream
from .pipeline import EdgePipeline
from .types import VehiclePass

logger = logging.getLogger("veritrack.edge")


class PassSink(Protocol):
    """Destination for completed vehicle passes."""

    def emit(self, vehicle_pass: VehiclePass, payload: bytes) -> None: ...

    def close(self) -> None: ...


class StdoutSink:
    """Newline-delimited JSON to stdout; useful for bench runs and piping."""

    def emit(self, vehicle_pass: VehiclePass, payload: bytes) -> None:
        sys.stdout.write(payload.decode("utf-8") + "\n")
        sys.stdout.flush()

    def close(self) -> None:
        sys.stdout.flush()


class KafkaSink:
    """Kafka producer sink keyed by ``pass_id``."""

    def __init__(self, brokers: str, topic: str, linger_ms: int = 20) -> None:
        try:
            from confluent_kafka import Producer
        except ImportError as exc:  # pragma: no cover - deployment dependent
            raise FatalEdgeError(
                "confluent-kafka is required for the kafka sink; install it or use --sink stdout"
            ) from exc
        self._topic = topic
        self._producer = Producer(
            {
                "bootstrap.servers": brokers,
                "linger.ms": linger_ms,
                "compression.type": "lz4",
                "enable.idempotence": True,
                "acks": "all",
                "message.max.bytes": 65536,
            }
        )

    @staticmethod
    def _on_delivery(err: Any, msg: Any) -> None:
        if err is not None:
            logger.error("kafka delivery failed for %s: %s", msg.key(), err)

    def emit(self, vehicle_pass: VehiclePass, payload: bytes) -> None:
        self._producer.produce(
            self._topic,
            key=vehicle_pass.pass_id.encode("ascii"),
            value=payload,
            callback=self._on_delivery,
        )
        self._producer.poll(0)

    def close(self) -> None:
        self._producer.flush(10.0)


def build_sink(name: str, brokers: Optional[str], topic: Optional[str]) -> PassSink:
    if name == "stdout":
        return StdoutSink()
    if name == "kafka":
        if not brokers or not topic:
            raise FatalEdgeError("--brokers and --topic are required for the kafka sink")
        return KafkaSink(brokers, topic)
    raise FatalEdgeError(f"Unknown sink '{name}'")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="veritrack-edge", description="VeriTrack edge node")
    parser.add_argument("--config", type=Path, required=True, help="Path to the JSON node config")
    parser.add_argument("--sink", default="stdout", choices=("stdout", "kafka"))
    parser.add_argument("--brokers", default=None, help="Kafka bootstrap servers")
    parser.add_argument("--topic", default=None, help="Kafka topic for raw passes")
    parser.add_argument("--stats-interval", type=float, default=60.0, help="Seconds between stat logs")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = run forever)")
    return parser.parse_args(argv)


def run(config: PipelineConfig, sink: PassSink, stats_interval: float, max_frames: int) -> int:
    pipeline = EdgePipeline(config)
    stream = VideoStream(config.stream)
    stop_requested = {"value": False}

    def _handle_signal(signum: int, _frame: Any) -> None:
        logger.info("signal %d received; draining", signum)
        stop_requested["value"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    last_stats = time.monotonic()
    processed = 0

    def _dispatch(passes: List[VehiclePass]) -> None:
        for vehicle_pass in passes:
            try:
                sink.emit(vehicle_pass, pipeline.payload_builder.serialize(vehicle_pass))
            except PayloadTooLargeError as exc:
                logger.error("dropping oversize pass: %s", exc)

    try:
        with stream:
            while not stop_requested["value"]:
                try:
                    frame = stream.read()
                except StreamError as exc:
                    logger.warning("%s", exc)
                    continue

                _dispatch(pipeline.process(frame))
                processed += 1

                now = time.monotonic()
                if stats_interval > 0.0 and now - last_stats >= stats_interval:
                    logger.info(
                        "stats %s dropped_frames=%d reconnects=%d",
                        pipeline.stats,
                        stream.dropped_frames,
                        stream.reconnect_count,
                    )
                    last_stats = now

                if max_frames and processed >= max_frames:
                    break

            _dispatch(pipeline.flush())
    finally:
        sink.close()

    logger.info("final stats %s", pipeline.stats)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
    except (OSError, ValueError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    configure_logging(config.log_level)
    logger.info("starting node=%s camera=%s", config.node_id, config.camera_id)

    try:
        sink = build_sink(args.sink, args.brokers, args.topic)
        return run(config, sink, args.stats_interval, args.max_frames)
    except FatalEdgeError as exc:
        logger.critical("fatal: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
