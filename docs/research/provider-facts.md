# Provider facts (checked 2026-10-30)

## A. Nova Sonic
- Operation `InvokeModelWithBidirectionalStream`; IDs `amazon.nova-2-sonic-v1:0`, `amazon.nova-2-5-sonic`. https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-2-5-sonic.html
- Python: sample uses `aws_sdk_bedrock_runtime` (smithy, experimental); boto3 only for setup. https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-getting-started.html
- Input LPCM 16-bit, 16 kHz, mono, base64. Output LPCM 24 kHz, 16-bit, mono. https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-getting-started.html
- Transcripts: `textOutput` events, role in `contentStart`; `generationStage` SPECULATIVE/FINAL in `additionalModelFields` (sample). Barge-in marker `{ "interrupted" : true }`: v1 doc only; 2.x not found. https://docs.aws.amazon.com/nova/latest/userguide/speech-bidirection.html
- Tools: `toolUse` out, toolResult in, `contentEnd` type `TOOL` triggers (v1 flow). 2.5 async tool calling supported; wire format not found. https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-2-5-sonic.html
- Session: 8-minute connection limit, continuation pattern in samples. Client ends with `promptEnd`, `sessionEnd`, input close. Idle timeout and server end signal: not found. https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-getting-started.html
- Quota: 20 concurrent sessions per account per Region. 2 card says not adjustable; 2.5 says ask account team. Service Quotas name: not found. https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-2-sonic.html
- API reference lists only `amazon.nova-sonic-v1:0` as accepted, conflicting with the cards. https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModelWithBidirectionalStream.html
- Pricing: Sonic rate rows on Bedrock pricing page not found. Third-party only: about $0.0045/min input (unverified). https://aws.amazon.com/bedrock/pricing/
- 2 to 2.5: context 1M to 256K; better reasoning, tool calling, instruction following; lower latency; polyglot voices; code-switching. https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-2-5-sonic.html

## B. OpenAI Realtime
- Names in guide: gpt-realtime-2.1 (example), gpt-realtime, -1.5, -2, -2.1-mini. GA status not found. https://developers.openai.com/api/docs/guides/realtime
- WebSocket URL: not found on fetched pages; transcription URL is `wss://api.openai.com/v1/realtime?intent=transcription`. https://developers.openai.com/api/docs/guides/realtime-transcription
- Audio: `audio/pcm` 24 kHz only; `audio/pcmu`, `audio/pcma`. https://developers.openai.com/api/reference/resources/realtime/client-events
- Client: `input_audio_buffer.append`, `.commit`, `.clear`; `response.create`, `response.cancel`, `conversation.item.truncate` (`audio_end_ms`). https://developers.openai.com/api/reference/resources/realtime/client-events
- Server: `response.output_audio.delta`; assistant text `response.output_audio_transcript.delta`; user transcript `conversation.item.input_audio_transcription.completed`. Function args `response.function_call_arguments.delta`; done event not found. https://developers.openai.com/api/docs/guides/realtime-transcription
- Transcription enable: `session.update` type `transcription`, `audio.input.transcription`. https://developers.openai.com/api/docs/guides/realtime-transcription
- Function results: `conversation.item.create` (`function_call_output`), then `response.create`. https://developers.openai.com/api/docs/guides/realtime-conversations
- Server VAD default for speech-to-speech: `threshold`, `prefix_padding_ms`, `silence_duration_ms`; no numeric defaults stated. `semantic_vad` eagerness `auto`=medium. https://developers.openai.com/api/docs/guides/realtime-vad
- Interruption: on `input_audio_buffer.speech_started` client stops playback and sends `conversation.item.truncate`; VAD-cancelled response shows `response.done` status `cancelled`; `response.cancel` is for push-to-talk. https://developers.openai.com/api/docs/guides/realtime-conversations
- Pricing per 1M tokens: gpt-realtime $32 audio in, $64 out; text $4 in, $16 out. gpt-realtime-2.1 text out $24. gpt-realtime-2.1-mini $10 / $20 audio. https://developers.openai.com/api/docs/pricing
- Per minute: gpt-live-transcribe and gpt-realtime-whisper $0.017/min. https://developers.openai.com/api/docs/pricing

## C. Synthetic input, echo, turn detection
- Neither vendor's fetched docs cover TTS or synthetic input accuracy, or echo and turn detection with machine speech: not found.

## D. Verified by CLI in the target account (2026-10-30, us-west-2)
- Foundation models listed ACTIVE: `amazon.nova-2-sonic-v1:0`, `amazon.nova-2-5-sonic`,
  `anthropic.claude-sonnet-5-5`, `anthropic.claude-haiku-5-5`, `anthropic.claude-opus-5-5`.
- Inference profiles present: `us.anthropic.claude-sonnet-5-5`, `us.anthropic.claude-haiku-5-5`,
  `us.anthropic.claude-opus-5-5`, and `global.` variants of each.
- Not yet verified: that `InvokeModelWithBidirectionalStream` accepts the 2.x Nova ids (the API
  reference lists only v1). The M1 spike settles it.
