#!/usr/bin/env python3
# coding=utf-8
from __future__ import annotations

import logging

from config_loader import build_args
from mic_engine import MicEngine
from web_app import IgnoreStateLogFilter, create_app


def main() -> None:
    args = build_args()
    if args.min_sec <= 0:
        raise SystemExit("--min-sec must be > 0")
    if args.max_sec < args.min_sec:
        raise SystemExit("--max-sec must be >= --min-sec")
    if args.cut_rate <= 0:
        raise SystemExit("--cut-rate must be > 0")
    if args.cut_tail_ms < 0:
        raise SystemExit("--cut-tail-ms must be >= 0")
    if args.voice_start_volume < 0:
        raise SystemExit("--voice-start-volume must be >= 0")
    if args.interupt_switch_delay < 0:
        raise SystemExit("--interupt-switch-delay must be >= 0")
    if args.interupt_early_release < 0:
        raise SystemExit("--interupt-early-release must be >= 0")
    if args.sample_rate <= 0:
        raise SystemExit("--sample-rate must be > 0")
    if args.block_ms <= 0:
        raise SystemExit("--block-ms must be > 0")
    if args.turn_detection_silence_ms <= 0:
        raise SystemExit("--turn-detection-silence-ms must be > 0")
    if args.long_silence_ms <= 0:
        raise SystemExit("--long-silence-ms must be > 0")
    if args.stutter_delay_ms < 0:
        raise SystemExit("--stutter-delay-ms must be >= 0")
    if args.stutter_delay_max_ms < 0:
        raise SystemExit("--stutter-delay-max-ms must be >= 0")
    if args.stutter_extend_mode != "hybrid_simple":
        raise SystemExit("--stutter-extend-mode currently only supports hybrid_simple")
    if not args.llm_base_url:
        raise SystemExit("--llm-base-url must not be empty")
    if not args.llm_model:
        raise SystemExit("--llm-model must not be empty")
    if args.llm_control_prompt_inject_threshold < 0:
        raise SystemExit("--llm-control-prompt-inject-threshold must be >= 0")
    if args.tts_enabled:
        if not args.tts_base_url:
            raise SystemExit("--tts-base-url must not be empty when TTS is enabled")
        if not args.tts_endpoint:
            raise SystemExit("--tts-endpoint must not be empty when TTS is enabled")
        if args.tts_group_sentences <= 0:
            raise SystemExit("--tts-group-sentences must be > 0")
        if args.tts_queue_ahead <= 0:
            raise SystemExit("--tts-queue-ahead must be > 0")
        if args.tts_num_step <= 0:
            raise SystemExit("--tts-num-step must be > 0")
        if args.tts_interupt_threshold < 0:
            raise SystemExit("--tts-interupt-threshold must be >= 0")
        if args.tts_interupt_volume < 0:
            raise SystemExit("--tts-interupt-volume must be >= 0")

    print("[QasrBasic] config:")
    print(f"  asr_base_url       = {args.asr_base_url}")
    print(f"  model              = {args.model}")
    print(f"  method             = {args.method}")
    print(f"  ui                 = http://{args.ui_host}:{args.ui_port}")
    print(f"  min_sec            = {args.min_sec}")
    print(f"  max_sec            = {args.max_sec}")
    print(f"  silence_ms         = {args.silence_ms}")
    print(f"  turn_asr_silence   = {args.turn_detection_silence_ms}")
    print(f"  long_silence_ms    = {args.long_silence_ms}")
    print(f"  stutter_extend     = {args.stutter_extend_enabled}")
    print(f"  stutter_mode       = {args.stutter_extend_mode}")
    print(f"  stutter_delay_ms   = {args.stutter_delay_ms}")
    print(f"  stutter_delay_max  = {args.stutter_delay_max_ms}")
    print(f"  phonetic_output    = {sorted(args.asr_phonetic_output_languages)}")
    print(f"  llm_base_url       = {args.llm_base_url}")
    print(f"  llm_endpoint       = {args.llm_endpoint}")
    print(f"  llm_model          = {args.llm_model}")
    print(f"  llm_stream         = {args.llm_stream}")
    print(f"  llm_persona_prompt = {len(args.llm_persona_prompt)} chars")
    print(f"  llm_tool_prompt    = {len(args.llm_tool_prompt)} chars")
    print(f"  llm_cp_threshold   = {args.llm_control_prompt_inject_threshold}")
    print(f"  llm_report_sec_id  = {args.llm_report_section_id}")
    print(f"  tts_engine         = omnivoice")
    print(f"  tts_enabled        = {args.tts_enabled}")
    print(f"  tts_base_url       = {args.tts_base_url}")
    print(f"  tts_endpoint       = {args.tts_endpoint}")
    print(f"  tts_voice          = {args.tts_voice}")
    print(f"  tts_language       = {args.tts_language}")
    print(f"  tts_instruct       = {args.tts_instruct!r}")
    print(f"  tts_speed          = {args.tts_speed}")
    print(f"  tts_duration       = {args.tts_duration}")
    print(f"  tts_num_step       = {args.tts_num_step}")
    print(f"  tts_guidance       = {args.tts_guidance_scale}")
    print(f"  tts_denoise        = {args.tts_denoise}")
    print(f"  tts_queue_ahead    = {args.tts_queue_ahead}")
    print(f"  tts_group_sent     = {args.tts_group_sentences}")
    print(f"  tts_interupt_thr   = {args.tts_interupt_threshold}")
    print(f"  tts_interupt_vol   = {args.tts_interupt_volume}")
    print(f"  tts_chop_except    = {sorted(args.tts_chop_exceptions)}")
    print(f"  tts_silence_types  = {sorted(args.tts_silence_types)}")
    print(f"  tts_packet_prefix  = {args.tts_packet_prefix!r}")
    print(f"  tts_packet_suffix  = {args.tts_packet_suffix!r}")
    print(f"  cut_rate           = {args.cut_rate}")
    print(f"  cut_tail_ms        = {args.cut_tail_ms}")
    print(f"  voice_start_volume = {args.voice_start_volume}")
    print(f"  interupt_early_rel = {args.interupt_early_release}")
    print(f"  interupt_sw_delay  = {args.interupt_switch_delay}")
    print(f"  sample_rate        = {args.sample_rate}")
    print(f"  block_ms           = {args.block_ms}")
    print(f"  echo_filter        = {args.echo_filter_enabled}")
    print(f"  interrupt_langs    = {sorted(args.interrupt_languages)}")
    print(f"  interrupt_prompt   = {len(args.interrupt_analyze_prompt)} chars")

    logging.getLogger("werkzeug").addFilter(IgnoreStateLogFilter())

    engine = MicEngine(args)
    app = create_app(args, engine)
    try:
        app.run(host=args.ui_host, port=args.ui_port, threaded=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
