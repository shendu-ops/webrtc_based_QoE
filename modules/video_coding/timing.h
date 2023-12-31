/*
 *  Copyright (c) 2011 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef MODULES_VIDEO_CODING_TIMING_H_
#define MODULES_VIDEO_CODING_TIMING_H_

#include <memory>

#include "absl/types/optional.h"
#include "api/video/video_timing.h"
#include "modules/video_coding/codec_timer.h"
#include "rtc_base/critical_section.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

class Clock;
class TimestampExtrapolator;

class VCMTiming {
 public:
  // The primary timing component should be passed
  // if this is the dual timing component.
  explicit VCMTiming(Clock* clock, VCMTiming* master_timing = NULL);
  virtual ~VCMTiming();

  // Resets the timing to the initial state.
  void Reset();

  // Set the amount of time needed to render an image. Defaults to 10 ms.
  void set_render_delay(int render_delay_ms);

  // Set the minimum time the video must be delayed on the receiver to
  // get the desired jitter buffer level.
  void SetJitterDelay(int required_delay_ms);

  // Set/get the minimum playout delay from capture to render in ms.
  void set_min_playout_delay(int min_playout_delay_ms);
  int min_playout_delay();

  // Set/get the maximum playout delay from capture to render in ms.
  void set_max_playout_delay(int max_playout_delay_ms);
  int max_playout_delay();

  // Increases or decreases the current delay to get closer to the target delay.
  // Calculates how long it has been since the previous call to this function,
  // and increases/decreases the delay in proportion to the time difference.
  void UpdateCurrentDelay(uint32_t frame_timestamp);

  // Increases or decreases the current delay to get closer to the target delay.
  // Given the actual decode time in ms and the render time in ms for a frame,
  // this function calculates how late the frame is and increases the delay
  // accordingly.
  void UpdateCurrentDelay(int64_t render_time_ms,
                          int64_t actual_decode_time_ms);

  // Stops the decoder timer, should be called when the decoder returns a frame
  // or when the decoded frame callback is called.
  void StopDecodeTimer(int32_t decode_time_ms, int64_t now_ms);
  // TODO(kron): Remove once downstream projects has been changed to use the
  // above function.
  void StopDecodeTimer(uint32_t time_stamp,
                       int32_t decode_time_ms,
                       int64_t now_ms,
                       int64_t render_time_ms);

  // Used to report that a frame is passed to decoding. Updates the timestamp
  // filter which is used to map between timestamps and receiver system time.
  void IncomingTimestamp(uint32_t time_stamp, int64_t last_packet_time_ms);

  // Returns the receiver system time when the frame with timestamp
  // |frame_timestamp| should be rendered, assuming that the system time
  // currently is |now_ms|.
  virtual int64_t RenderTimeMs(uint32_t frame_timestamp, int64_t now_ms); //remove const

  // Returns the maximum time in ms that we can wait for a frame to become
  // complete before we must pass it to the decoder.
  virtual int64_t MaxWaitingTime(int64_t render_time_ms, int64_t now_ms) const;

  // Returns the current target delay which is required delay + decode time +
  // render delay.
  int TargetVideoDelay() const;

  // Return current timing information. Returns true if the first frame has been
  // decoded, false otherwise.
  virtual bool GetTimings(int* max_decode_ms,
                          int* current_delay_ms,
                          int* target_delay_ms,
                          int* jitter_buffer_ms,
                          int* min_playout_delay_ms,
                          int* render_delay_ms) const;

  void SetTimingFrameInfo(const TimingFrameInfo& info);
  absl::optional<TimingFrameInfo> GetTimingFrameInfo();

  //add
  //////////////////////////////////////////////
  bool readfile_flag_ = true;
  int delay_self_ = 0;
  //////////////////////////////////////////////

  enum { kDefaultRenderDelayMs = 10 };
  enum { kDelayMaxChangeMsPerS = 100 };

 protected:
  int RequiredDecodeTimeMs() const RTC_EXCLUSIVE_LOCKS_REQUIRED(crit_sect_);
  int64_t RenderTimeMsInternal(uint32_t frame_timestamp, int64_t now_ms) const
      RTC_EXCLUSIVE_LOCKS_REQUIRED(crit_sect_);
  
  // add
  ///////////////////////////////////////////////////////////////////////////////////////
  int64_t RenderTimeMsInternal_Self(uint32_t frame_timestamp, int64_t now_ms)
      RTC_EXCLUSIVE_LOCKS_REQUIRED(crit_sect_);
  ///////////////////////////////////////////////////////////////////////////////////////

  int TargetDelayInternal() const RTC_EXCLUSIVE_LOCKS_REQUIRED(crit_sect_);

 private:
  rtc::CriticalSection crit_sect_;
  Clock* const clock_;
  bool master_ RTC_GUARDED_BY(crit_sect_);
  TimestampExtrapolator* ts_extrapolator_ RTC_GUARDED_BY(crit_sect_);
  std::unique_ptr<VCMCodecTimer> codec_timer_ RTC_GUARDED_BY(crit_sect_);
  int render_delay_ms_ RTC_GUARDED_BY(crit_sect_);
  // Best-effort playout delay range for frames from capture to render.
  // The receiver tries to keep the delay between |min_playout_delay_ms_|
  // and |max_playout_delay_ms_| taking the network jitter into account.
  // A special case is where min_playout_delay_ms_ = max_playout_delay_ms_ = 0,
  // in which case the receiver tries to play the frames as they arrive.
  int min_playout_delay_ms_ RTC_GUARDED_BY(crit_sect_);
  int max_playout_delay_ms_ RTC_GUARDED_BY(crit_sect_);
  int jitter_delay_ms_ RTC_GUARDED_BY(crit_sect_);
  int current_delay_ms_ RTC_GUARDED_BY(crit_sect_);
  uint32_t prev_frame_timestamp_ RTC_GUARDED_BY(crit_sect_);
  absl::optional<TimingFrameInfo> timing_frame_info_ RTC_GUARDED_BY(crit_sect_);
  size_t num_decoded_frames_ RTC_GUARDED_BY(crit_sect_);
};
}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_TIMING_H_
