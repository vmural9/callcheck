# Nova Sonic spike: observed wire behaviour

Observed 2026-10-30 against the target account in us-west-2, with `spikes/nova_sonic_spike.py`
(throwaway) and `aws-sdk-bedrock-runtime` 0.12.0 (smithy, experimental). Raw event logs were
written to `spikes/out/*.jsonl` (git-ignored, not committed). Everything below was seen on the
wire unless it says "not observed". Timestamps are seconds since the session opened.

Evidence base: 8 live sessions (probe, tool, hears x2 [one synthesizer, one listener],
bargein x2, model-id check, plus one probe run whose output was discarded because the
receiver mis-keyed output events: they name content by `contentId`, not `contentName`).
Every audio payload is elided as `"<b64>"` below.

## 1. Model id

`InvokeModelWithBidirectionalStream` in us-west-2 accepted:

| Model id | Result |
|---|---|
| `amazon.nova-2-5-sonic` | accepted, used for every observation below |
| `amazon.nova-2-sonic-v1:0` | accepted (one short text-turn session; same event shapes, replied and ended normally) |
| `amazon.nova-sonic-v1:0` | not tried: ids were tried in the order above and the first was accepted |

Chosen: `amazon.nova-2-5-sonic`, because it was first in the preference order and worked. The API
reference lists only the v1 id as valid; that is out of date, the 2.x ids work. A rejected id was
never seen, so its failure shape is not observed: it is not known whether a rejection raises at
the invoke call or on the first `receive()`.

A plain model id (no region prefix, no inference profile) was accepted as the `model_id`. Whether
a regional or global inference-profile id is also accepted was not tried.

## 2. Event sequence (client to model)

Everything is a JSON object `{"event": {<name>: {...}}}` inside one stream chunk. This exact
order worked, with no audio streamed before the first text turn:

```
sessionStart            {inferenceConfiguration:{maxTokens,topP,temperature},
                         turnDetectionConfiguration:{endpointingSensitivity:"MEDIUM"}}
promptStart             {promptName, textOutputConfiguration:{mediaType:"text/plain"},
                         audioOutputConfiguration:{mediaType:"audio/lpcm",sampleRateHertz:24000,
                         sampleSizeBits:16,channelCount:1,voiceId:"matthew",encoding:"base64",
                         audioType:"SPEECH"},
                         [toolUseOutputConfiguration:{mediaType:"application/json"},
                          toolConfiguration:{tools:[...]}]}
contentStart  SYSTEM    {promptName, contentName, type:"TEXT", interactive:false, role:"SYSTEM",
                         textInputConfiguration:{mediaType:"text/plain"}}
textInput               {promptName, contentName, content:"<system prompt>"}
contentEnd              {promptName, contentName}
contentStart  AUDIO     {promptName, contentName, type:"AUDIO", interactive:true, role:"USER",
                         audioInputConfiguration:{mediaType:"audio/lpcm",sampleRateHertz:16000,
                         sampleSizeBits:16,channelCount:1,audioType:"SPEECH",encoding:"base64"}}
  audioInput  (repeated){promptName, contentName, content:"<b64 PCM16 16 kHz mono>"}
  ... text turn, tool result, etc. interleave here while the audio content stays open ...
contentEnd              {promptName, contentName}      (the audio content)
promptEnd               {promptName}
sessionEnd              {}
(then close the input stream)
```

Text input turn (same shape as the system prompt, role `USER`, `interactive:true`), sent while
the AUDIO content was open:

```json
{"event":{"contentStart":{"promptName":"<p>","contentName":"<c>","type":"TEXT","interactive":true,"role":"USER","textInputConfiguration":{"mediaType":"text/plain"}}}}
{"event":{"textInput":{"promptName":"<p>","contentName":"<c>","content":"my first name is Maria"}}}
{"event":{"contentEnd":{"promptName":"<p>","contentName":"<c>"}}}
```

Facts about the input side:

- The server sends no acknowledgement to any input event (no ack for `sessionStart`,
  `promptStart`, content blocks). The first server event is a `usageEvent` about 0.8 s after
  the system prompt (see 7). No error occurred with the sequence above; the failure
  mode for a malformed event or bad model id was not observed.
- `audioInput` frames: 512 samples (1024 bytes, 32 ms) at real-time pace. A text turn was
  accepted with no audio frames sent for the first 3 s of the session.
- The audio `contentStart` was opened before any audio and kept open for the whole session; a
  single AUDIO content carried every utterance. Closing and reopening AUDIO contents between
  utterances was not tried.
- The `audioInput` events are not echoed. Neither are text turns: a `USER` text turn produces no
  `textOutput` with role `USER`.

## 3. Output events (model to client)

Output events carry `completionId`, `promptName`, `sessionId` and, for content events,
`contentId` (a server-generated id, not the client's `contentName`). One response looked like
this (probe, text turn `(call connected)`):

```
3.558 completionStart                                      {completionId,promptName,sessionId}
3.559 contentStart  ASSISTANT TEXT  stage SPECULATIVE      textOutput "Hello, how can I help you today?"
3.559 contentEnd    TEXT  stopReason PARTIAL_TURN
3.683 contentStart  ASSISTANT AUDIO                        audioOutput x22 (5120 b64 chars each)
4.782 contentEnd    AUDIO stopReason END_TURN
5.828 contentStart  ASSISTANT TEXT  stage FINAL            textOutput "Hello, how can I help you today?"
5.829 contentEnd    TEXT  stopReason END_TURN
```

Shapes:

```json
{"completionStart":{"completionId":"…","promptName":"…","sessionId":"…"}}
{"contentStart":{"additionalModelFields":"{\"generationStage\":\"SPECULATIVE\"}","completionId":"…","contentId":"…","promptName":"…","role":"ASSISTANT","sessionId":"…","textOutputConfiguration":{"mediaType":"text/plain"},"type":"TEXT"}}
{"textOutput":{"completionId":"…","content":"Hello, how can I help you today?","contentId":"…","promptName":"…","role":"ASSISTANT","sessionId":"…"}}
{"contentEnd":{"completionId":"…","contentId":"…","promptName":"…","sessionId":"…","stopReason":"PARTIAL_TURN","type":"TEXT"}}
{"contentStart":{"audioOutputConfiguration":{"channelCount":1,"encoding":"base64","mediaType":"audio/lpcm","sampleRateHertz":24000,"sampleSizeBits":16},"completionId":"…","contentId":"…","promptName":"…","role":"ASSISTANT","sessionId":"…","type":"AUDIO"}}
{"audioOutput":{"completionId":"…","content":"<b64>","contentId":"…","promptName":"…","role":"ASSISTANT","sessionId":"…"}}
{"contentEnd":{"…":"…","stopReason":"END_TURN","type":"AUDIO"}}
```

- Audio: 24 kHz PCM16 mono, base64. Chunks are 5120 base64 characters = 3840 bytes = 80 ms
  (one chunk of 12800 characters = 200 ms was also seen, so do not assume a fixed size). The
  AUDIO `contentStart` has no `additionalModelFields` and no stage.
- Assistant text appears twice per spoken turn: a `SPECULATIVE` TEXT content sent right before
  the audio (the text of what the model is about to say), and a `FINAL` TEXT content sent after
  the audio content ends (1.0 to 1.5 s later, apparently when playback would be done). The two
  carried identical text when nothing interrupted. The stage is a JSON string inside
  `additionalModelFields`, in `contentStart` only; `textOutput` carries no stage.
- A long reply is split into several `SPECULATIVE` TEXT + AUDIO pairs, each closed with
  `stopReason: "PARTIAL_TURN"`; the last one closes with `END_TURN`. In the counting runs each
  pair was about 2 s of speech. Those runs ended in an interruption, and then the `FINAL` text
  came as one TEXT content per pair; an uninterrupted multi-pair reply was not observed, so how
  its `FINAL` text is chunked is unknown.
- User transcript: a TEXT content with `role: "USER"` and `generationStage` `FINAL` only (no
  speculative user text). Observed (idle model, item 9; `role` is present on `contentStart`
  and on `textOutput`, the stage only on `contentStart`):

  ```json
  {"contentStart":{"additionalModelFields":"{\"generationStage\":\"FINAL\"}","completionId":"…","contentId":"…","promptName":"…","role":"USER","sessionId":"…","textOutputConfiguration":{"mediaType":"text/plain"},"type":"TEXT"}}
  {"textOutput":{"completionId":"…","content":"hello, can you hear me? testing one two three.","contentId":"…","promptName":"…","role":"USER","sessionId":"…"}}
  {"contentEnd":{"completionId":"…","contentId":"…","promptName":"…","sessionId":"…","stopReason":"PARTIAL_TURN","type":"TEXT"}}
  ```
 Its `contentEnd` has `stopReason: "PARTIAL_TURN"`. With an idle model the transcript
  arrived 0.55 s after the speech audio ended (see 9).
- `completionEnd` (`{completionId,promptName,sessionId,stopReason:"END_TURN"}`) was not seen
  after a normal turn; it appeared only after the client sent `promptEnd`/`sessionEnd` (see 6).
  Do not use it as a turn boundary. The turn boundary is the AUDIO `contentEnd` with
  `stopReason: "END_TURN"`.
- Two more output events, not in the AWS documentation reviewed beforehand: `userSpeechStart`
  `{inputAudioOffsetMs,promptName,sessionId}` and `userSpeechEnd`
  `{inputAudioDetectionOffsetMs,inputAudioOffsetMs,promptName,sessionId}`. The offsets are
  milliseconds on the client's input audio timeline.
- A 2-second 440 Hz tone followed by 3 s of silence produced no events of any kind (no
  `userSpeechStart`, no transcript, no usage). Synthetic non-speech does not count as speech.

## 4. Tool use

Prompt start carried the tool list (`inputSchema.json` is a JSON-encoded string, not an object):

```json
"toolUseOutputConfiguration":{"mediaType":"application/json"},
"toolConfiguration":{"tools":[{"toolSpec":{"name":"save_field","description":"Save one field the caller has told us.","inputSchema":{"json":"{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"value\":{\"type\":\"string\"}},\"required\":[\"name\",\"value\"]}"}}}]}
```

System prompt asked the model to call `save_field` when it learned the first name; text turn:
`my first name is Maria`. Observed:

```
3.638 contentStart  role TOOL, type TOOL, toolUseOutputConfiguration{mediaType:application/json}
3.638 toolUse       {"content":"{\"name\":\"first_name\",\"value\":\"Maria\"}","contentId":"…",
                     "role":"TOOL","toolName":"save_field","toolUseId":"chatcmpl-tool-83850decee808397",…}
3.639 contentEnd    {stopReason:"TOOL_USE", type:"TOOL"}
```

`toolUse.content` is a JSON string holding the tool arguments (here the model chose the argument
names `name` and `value` of the test schema: `name:"first_name"`, `value:"Maria"`). The tool call came
0.6 s after the text turn and before any speech. Result sent back (immediately after the
`contentEnd`, 1 ms later):

```json
{"event":{"contentStart":{"promptName":"<p>","contentName":"<new>","interactive":false,"type":"TOOL","role":"TOOL","toolResultInputConfiguration":{"toolUseId":"chatcmpl-tool-83850decee808397","type":"TEXT","textInputConfiguration":{"mediaType":"text/plain"}}}}}
{"event":{"toolResult":{"promptName":"<p>","contentName":"<new>","content":"{\"saved\": true}"}}}
{"event":{"contentEnd":{"promptName":"<p>","contentName":"<new>"}}}
```

The model continued: 0.4 s later it sent SPECULATIVE text `I have saved your first name as Maria.`,
then audio (4.155 to 5.468), then the FINAL text. It did not speak before calling the tool.
Not observed: several tool calls in one turn, a tool error result, a slow tool (async calling),
and sending the result before the `TOOL_USE` `contentEnd` arrived.

## 5. Barge-in

Setup: system prompt "count aloud from one to N, one number per sentence"; text turn asking for
it; then 2.84 s of synthesized speech ("hello, can you hear me, testing one two three", item 9)
sent as audio input while the model was speaking. Two runs:

| Run | Speech sent (wall) | `userSpeechStart` received | `userSpeechEnd` `inputAudioDetectionOffsetMs` | `userSpeechEnd` received |
|---|---|---|---|---|
| count to 30, speech at 6.9 s | 6.9 to 9.7 s | 15.006 s | 6520 | 15.007 s |
| count to 60, speech at 5.4 s | 5.4 to 8.2 s | 10.614 s | 5280 | 10.616 s |

Compare the idle model (item 9): speech start event 0.6 s after the audio began, end event 0.4 s
after the speech audio had been sent. During speech by the model the user-speech events were
late by 2.3 s (run 2) and 5.5 s (run 1) after the audio they refer to had been sent, and
`userSpeechStart`, `userSpeechEnd` and the user transcript all arrived within 3 ms of each
other.

The interruption signal, in an ASSISTANT TEXT content (FINAL stage):

```json
{"contentStart":{"additionalModelFields":"{\"generationStage\":\"FINAL\"}","role":"ASSISTANT","type":"TEXT",…}}
{"textOutput":{"content":"{ \"interrupted\" : true }","role":"ASSISTANT",…}}
{"contentEnd":{"stopReason":"INTERRUPTED","type":"TEXT",…}}
```

Sequence at the interruption (run 2, 10.614 s): `userSpeechStart`; the in-flight AUDIO
`contentEnd` with `stopReason: "PARTIAL_TURN"`; the FINAL TEXT block(s) for what was said,
including the `interrupted` text and its `contentEnd` `stopReason: "INTERRUPTED"`; then
`userSpeechEnd`, the USER transcript `hello? can you hear me? testing one, two, three.`, and a
new assistant response `Yes, I can hear you. How can I help you today?`. The model abandoned
the count (run 2 stopped after 18 numbers of 60) and answered the interrupting speech. The
position of the `interrupted` block relative to the FINAL text blocks differed between runs
(after "One. Two... Six." in run 1, before it in run 2), so detect it by content or by
`stopReason: "INTERRUPTED"`, not by position.

Important for a playback client: the model sends audio ahead of real time. In run 2 it had sent
126 chunks (about 10 s of speech) by 10.6 s, having started at 3.8 s, so about 1.5x real time.
The client must discard its unplayed audio buffer when the interruption arrives. Why the user
events lagged 2 to 5 s while the model was speaking is not determined (candidate: the server
reports them only at an output-block boundary or after its own output pacing; the data
distinguishes neither).

## 6. End signals and idle

Client close: the sequence `contentEnd` (audio) -> `promptEnd` -> `sessionEnd` -> input stream
close produced, 0.3 to 0.5 s later (every session that was closed cleanly, six of them):

```json
{"completionEnd":{"completionId":"…","promptName":"…","sessionId":"…","stopReason":"END_TURN"}}
```

followed immediately by the receiver returning `None` (the smithy `receive()` returns `None`;
no exception). This is the clean end signal.

Idle for 60 s: after the last client input at 17.9 s the client sent nothing. At 73.196 s (55.3 s after
the last input) the output stream raised, instead of a normal event:

```
ValidationException(fault='client', message='RequestId=<sessionId> : Timed out waiting for audio bytes or interactive content. Please ensure gaps between audio bytes and interactive content are less than 55 seconds.')
```

The stream ends there. Sends after it did not raise at the call site; they were silently
useless. The 55 s was measured from the last client input; the model was not speaking during
that window, so whether model output resets the timer was not tested. A client that pauses
audio must keep sending silence frames, or a text turn, at least every 55 s. The documented
session limit of 8 minutes was not tested.

## 7. `usageEvent`

Sent unprompted, many times per session (10 to 33 per session), at the start (system prompt
tokens), after each recognised speech or text input and after each output block (the tone input
of item 3 produced none):

```json
{"usageEvent":{"completionId":"…","promptName":"…","sessionId":"…",
 "details":{"delta":{"input":{"speechTokens":0,"textTokens":22},"output":{"speechTokens":0,"textTokens":0}},
            "total":{"input":{"speechTokens":0,"textTokens":22},"output":{"speechTokens":0,"textTokens":0}}},
 "totalInputTokens":22,"totalOutputTokens":0,"totalTokens":22}}
```

`delta` is the change since the previous usage event; `total` is cumulative for the session.
`totalTokens = totalInputTokens + totalOutputTokens`. Output speech tokens are counted as
audio is generated: about 23 per second of speech (40 for the 1.76 s greeting). At the end of the 30-number
count, total input was 1106 and total output 556 tokens. A session with no audio sent already
reports 150 input speech tokens within 1 s of the system prompt (before any audio frame was
sent), so that count is not audio duration. The usage event is the only cost data on the wire.

## 8. Does a text user turn make the model speak first?

Yes. After the system prompt and the open audio content, the text turn `(call connected)` sent
at 3.0 s produced a greeting starting 0.54 s later: SPECULATIVE text "Hello, how can I help you
today?", then 22 audio chunks (about 1.8 s). The AUDIO content was open but no audio frame had
been sent yet. The 2.0 id (`amazon.nova-2-sonic-v1:0`) behaved the same ("Hello! Thanks for
calling. How can I help you today?"). Without a text turn the model said nothing in the 3 s
that followed the system prompt in every session (seen in the first 3 s of each run; no
separate no-text session was run, so a longer silence might behave differently).

## 9. Does Nova hear Nova?

Yes. A second session (system prompt "voice reader", text turn
`Say: hello, can you hear me, testing one two three`) spoke exactly that sentence (assistant
text identical, 136 320 bytes = 2.84 s of 24 kHz audio, captured from the single AUDIO
content). It was resampled with `soxr.resample(int16, 24000, 16000)` and fed to a fresh session
at real-time pace, followed by 2 s of silence frames:

```
3.605 userSpeechStart  {inputAudioOffsetMs:0}
6.412 USER TEXT FINAL  textOutput "hello, can you hear me? testing one two three."
6.562 userSpeechEnd    {inputAudioDetectionOffsetMs:3120, inputAudioOffsetMs:2600}
6.751 assistant        "Yes, I can hear you clearly."
```

All 9 test words were recognised (`hello`, `can`, `you`, `hear`, `me`, `testing`, `one`, `two`,
`three`); the recogniser added punctuation. The user transcript arrived 0.55 s after the end of
the speech samples, while the 2 s of trailing silence was still being sent. The same speech fed
during barge-in (item 5) produced the near-identical transcript `hello? can you hear me?
testing one, two, three.`

## 10. Credentials

The smithy client has no boto3 hook; it resolves its own credentials. What worked in 0.12.0:

```python
creds = boto3.Session(profile_name=os.environ["AWS_PROFILE"]).get_credentials().get_frozen_credentials()
config = await AsyncBedrockRuntimeConfig.resolve(
    region=os.environ["AWS_REGION"],
    aws_access_key_id=creds.access_key,
    aws_secret_access_key=creds.secret_key,
    aws_session_token=creds.token,
    transport=AWSCRTHTTPClient(),          # from smithy_http.aio.crt
)
client = AsyncBedrockRuntimeClient(config=config)
stream = await client.invoke_model_with_bidirectional_stream(
    InvokeModelWithBidirectionalStreamOperationInput(model_id=model_id))
```

Differences from the AWS sample code (written for an earlier release):

- The client and config classes are `AsyncBedrockRuntimeClient` and `AsyncBedrockRuntimeConfig`
  (`.resolve(...)` is a coroutine); `BedrockRuntimeClient` and `Config` are not present in the
  installed 0.12.0 modules.
- The default transport (aiohttp) raises `UnsupportedTransportError: ... does not support duplex
  (bidirectional) event streaming`. A CRT transport is mandatory, which needs the `awscrt`
  package: the optional dependency is now `aws-sdk-bedrock-runtime[awscrt]` (pyproject changed
  accordingly; awscrt 0.32.2 installed).
- `client.invoke_model_with_bidirectional_stream(...)` returns a `DuplexEventStream` immediately;
  send with `stream.input_stream.send(InvokeModelWithBidirectionalStreamInputChunk(value=
  BidirectionalInputPayloadPart(bytes_=json_bytes)))`; call `await stream.await_output()` to get
  `(output, receiver)` once, then `await receiver.receive()` in a loop (`None` = end). Each
  received item has `.value.bytes_`; error variants (`ValidationException` etc.) raise from
  `receive()`.
- Not tried: `AsyncBedrockRuntimeConfig.resolve(profile=...)`, which the source suggests would
  read the profile itself. Frozen credentials do not refresh; a session using temporary
  credentials that expire mid-session was not exercised.

## What the client must do

1. Build the config as in item 10, with the CRT transport and the `bedrock` extra including
   `awscrt`. Model id `amazon.nova-2-5-sonic` in us-west-2, from config. Both it and
   `amazon.nova-2-sonic-v1:0` were accepted, but a rejection was never observed, so the
   client cannot yet say whether a bad id raises at the invoke call or on the first `receive()`.
   If a fallback to the 2.0 id is wanted, catch both places; the spike's own detection (stream
   ended before any event arrived) is a guess, not a verified signal.
2. Open order: `sessionStart`, `promptStart` (with `toolConfiguration` if tools are used),
   SYSTEM text block, then the AUDIO `contentStart` held open for the whole session. Generate
   UUIDs for `promptName` and each `contentName`.
3. Stream caller audio as 16 kHz mono PCM16, base64, in real time (32 ms frames worked). Send
   silence frames while there is nothing to say: the server times out 55 s after the last
   client audio or interactive content, with a `ValidationException` on the output stream.
4. To make the model speak first, send a USER text turn (`interactive:true`), e.g.
   `(call connected)`, after the open handshake; it replied in about 0.5 s. A text turn in the
   middle of a session works the same way.
5. Read events by their single key under `event`. Build the transcript from `textOutput` by
   `role` of the owning `contentStart` (look it up by `contentId`); keep `FINAL` stage text for the
   transcript, but drop any TEXT block whose `contentEnd` has `stopReason: "INTERRUPTED"` (it
   holds the marker `{ "interrupted" : true }`, not speech). Use the `SPECULATIVE` text if a
   live caption is wanted. USER content has only a `FINAL` stage.
6. Decode `audioOutput.content` as 24 kHz PCM16 mono; chunk size varies (80 ms and 200 ms
   were seen). Treat AUDIO `contentEnd` with `stopReason` `END_TURN` as end of a reply;
   `PARTIAL_TURN` usually means more audio follows, but an interrupted reply also ends its
   in-flight AUDIO content with `PARTIAL_TURN` and no more audio follows (see rule 8): do not
   wait for `END_TURN` or further audio of an interrupted reply. Do not wait for `completionEnd`
   between turns.
7. Tool calls: on `toolUse` (key fields `toolName`, `toolUseId`, `content` = JSON string of the
   arguments), run the tool, then send a TOOL `contentStart` with `toolResultInputConfiguration
   {toolUseId,type:"TEXT",textInputConfiguration}`, a `toolResult` with a JSON string in
   `content`, and `contentEnd`. The model then speaks a confirmation by itself.
8. Barge-in: when an ASSISTANT TEXT content ends with `stopReason: "INTERRUPTED"` (text
   `{ "interrupted" : true }`), flush any queued playback audio; the server stops generating the
   rest of the old reply and responds to the interrupting speech. Allow 2 to 5 s of lag between
   the interrupting audio and these events while the model is speaking.
9. `userSpeechStart`/`userSpeechEnd` events give offsets on the client's input timeline
   (`inputAudioOffsetMs`); use them to time user turns, not wall-clock arrival.
10. Cost: read `usageEvent.details.total` and `totalTokens` (cumulative); speech and text tokens
    are counted separately for input and output.
11. Clean end: send `contentEnd` (audio), `promptEnd`, `sessionEnd`, then close the input
    stream; read until `receive()` returns `None` (a final `completionEnd` arrives first).
    Wait at least 1 s for it before tearing down.
12. Catch exceptions from `receive()` and treat them as the end of the session; sends after a
    server-side failure do not raise.
13. Sessions are capped at 8 minutes by the service (documented, not tested): plan a
    continuation or a hard stop before that.

## Not observed

- The id `amazon.nova-sonic-v1:0`, and the shape of an id rejection.
- `endpointingSensitivity` other than `MEDIUM`; voices other than `matthew`.
- The 8-minute session limit; the concurrent-session quota; throttling.
- Multiple tool calls in one turn, tool errors, async tool calls.
- Closing and reopening the AUDIO content between utterances.
- The cause of the 2 to 5 s lag of user-speech events during model speech (item 5).
- Credential refresh in a long session; `resolve(profile=...)`.
