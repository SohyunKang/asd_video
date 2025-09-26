# asd_video

Input Video:  data/IF2001_2_1_1024080292_0.mp4
fps: 30.01,frame_count: 186
---> Data Load Time: 0.2478
호명 타깃:  가온아
Start: 1.67
---> Time Segment Load Time: 0.0004
124 1 4
---> Face Recognition Time: 13.1050

2025-09-26 16:08:31.177001: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-09-26 16:08:31.228971: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 AVX512_FP16 AVX_VNNI AMX_TILE AMX_INT8 AMX_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-09-26 16:08:32.348602: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-09-26 16:08:33.491156: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:47] Overriding orig_value setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1758870513.492262 3168038 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 70434 MB memory:  -> device: 0, name: NVIDIA H100 PCIe, pci bus id: 0000:2a:00.0, compute capability: 9.0
2025-09-26 16:08:36.162912: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:473] Loaded cuDNN version 91300
E0000 00:00:1758870517.195873 3168446 ptx_compiler_helpers.cc:88] *** WARNING *** Invoking ptxas with version 12.1.105, which corresponds to a CUDA version <=12.6.2. CUDA versions 12.x.y up to and including 12.6.2 miscompile certain edge cases around clamping.
Please upgrade to CUDA 12.6.3 or newer.

---> Face Recog Data Load Time: 0.0004
cuda
---> Gazelle Model Load Time: 6.7684
---> Gazelle Inference Time: 6.3652
반응 End: 2.07
---> Visualization 1 Time: 6.2129
---> Visualization 2 Time: 61.3610
****** After Time ******: 74.7877

Using cache found in /home/sohyunkang/.cache/torch/hub/fkryan_gazelle_main
Using cache found in /home/sohyunkang/.cache/torch/hub/facebookresearch_dinov2_main
/home/sohyunkang/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/swiglu_ffn.py:51: UserWarning: xFormers is not available (SwiGLU)
  warnings.warn("xFormers is not available (SwiGLU)")
/home/sohyunkang/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:33: UserWarning: xFormers is not available (Attention)
  warnings.warn("xFormers is not available (Attention)")
/home/sohyunkang/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py:40: UserWarning: xFormers is not available (Block)
  warnings.warn("xFormers is not available (Block)")
ffmpeg version 4.4.1-static https://johnvansickle.com/ffmpeg/  Copyright (c) 2000-2021 the FFmpeg developers
  built with gcc 8 (Debian 8.3.0-6)
  configuration: --enable-gpl --enable-version3 --enable-static --disable-debug --disable-ffplay --disable-indev=sndio --disable-outdev=sndio --cc=gcc --enable-fontconfig --enable-frei0r --enable-gnutls --enable-gmp --enable-libgme --enable-gray --enable-libaom --enable-libfribidi --enable-libass --enable-libvmaf --enable-libfreetype --enable-libmp3lame --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-librubberband --enable-libsoxr --enable-libspeex --enable-libsrt --enable-libvorbis --enable-libopus --enable-libtheora --enable-libvidstab --enable-libvo-amrwbenc --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libdav1d --enable-libxvid --enable-libzvbi --enable-libzimg
  libavutil      56. 70.100 / 56. 70.100
  libavcodec     58.134.100 / 58.134.100
  libavformat    58. 76.100 / 58. 76.100
  libavdevice    58. 13.100 / 58. 13.100
  libavfilter     7.110.100 /  7.110.100
  libswscale      5.  9.100 /  5.  9.100
  libswresample   3.  9.100 /  3.  9.100
  libpostproc    55.  9.100 / 55.  9.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'data/IF2001_2_1_1024080292_0.mp4':
  Metadata:
    major_brand     : qt  
    minor_version   : 0
    compatible_brands: qt  
    creation_time   : 2024-08-11T11:49:05.000000Z
  Duration: 00:00:06.20, start: 0.000000, bitrate: 4970 kb/s
  Stream #0:0(und): Video: hevc (Main) (hvc1 / 0x31637668), yuv420p(tv, bt709), 1280x720, 4865 kb/s, 30.01 fps, 30 tbr, 600 tbn, 600 tbc (default)
    Metadata:
      creation_time   : 2024-08-11T11:49:05.000000Z
      handler_name    : Core Media Video
      vendor_id       : [0][0][0][0]
      encoder         : HEVC
  Stream #0:1(und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, mono, fltp, 94 kb/s (default)
    Metadata:
      creation_time   : 2024-08-11T11:49:05.000000Z
      handler_name    : Core Media Audio
      vendor_id       : [0][0][0][0]
Input #1, image2, from 'data/IF2001_2_1_1024080292_0_overlay_frames/frame_%05d.png':
  Duration: 00:00:06.20, start: 0.000000, bitrate: N/A
  Stream #1:0: Video: png, rgb24(pc), 1280x720, 30.01 fps, 30.01 tbr, 30.01 tbn, 30.01 tbc
Stream mapping:
  Stream #0:0 (hevc) -> overlay:main
  Stream #1:0 (png) -> format
  format -> Stream #0:0 (libx264)
  Stream #0:1 -> #0:1 (copy)
Press [q] to stop, [?] for help
[image2 @ 0x180b2240] Thread message queue blocking; consider raising the thread_queue_size option (current value: 8)
[libx264 @ 0x180c20c0] using cpu capabilities: MMX2 SSE2Fast SSSE3 SSE4.2 AVX FMA3 BMI2 AVX2 AVX512
[libx264 @ 0x180c20c0] profile High, level 3.1, 4:2:0, 8-bit
[libx264 @ 0x180c20c0] 264 - core 164 r3075 66a5bc1 - H.264/MPEG-4 AVC codec - Copyleft 2003-2021 - http://www.videolan.org/x264.html - options: cabac=1 ref=3 deblock=1:0:0 analyse=0x3:0x113 me=hex subme=7 psy=1 psy_rd=1.00:0.00 mixed_ref=1 me_range=16 chroma_me=1 trellis=1 8x8dct=1 cqm=0 deadzone=21,11 fast_pskip=1 chroma_qp_offset=-2 threads=22 lookahead_threads=3 sliced_threads=0 nr=0 decimate=1 interlaced=0 bluray_compat=0 constrained_intra=0 bframes=3 b_pyramid=2 b_adapt=1 b_bias=0 direct=1 weightb=1 open_gop=0 weightp=2 keyint=250 keyint_min=25 scenecut=40 intra_refresh=0 rc_lookahead=40 rc=crf mbtree=1 crf=23.0 qcomp=0.60 qpmin=0 qpmax=69 qpstep=4 ip_ratio=1.40 aq=1:1.00
Output #0, mp4, to 'results/IF2001_2_1_1024080292_0_gaze_sound.mp4':
  Metadata:
    major_brand     : qt  
    minor_version   : 0
    compatible_brands: qt  
    encoder         : Lavf58.76.100
  Stream #0:0: Video: h264 (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720, q=2-31, 30 fps, 15360 tbn (default)
    Metadata:
      encoder         : Lavc58.134.100 libx264
    Side data:
      cpb: bitrate max/min/avg: 0/0/0 buffer size: 0 vbv_delay: N/A
  Stream #0:1(und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, mono, fltp, 94 kb/s (default)
    Metadata:
      creation_time   : 2024-08-11T11:49:05.000000Z
      handler_name    : Core Media Audio
      vendor_id       : [0][0][0][0]
frame=    1 fps=0.0 q=0.0 size=       0kB time=00:00:01.48 bitrate=   0.3kbits/s speed=21.6x    
frame=   81 fps=0.0 q=29.0 size=       0kB time=00:00:04.15 bitrate=   0.1kbits/s speed=7.28x    
frame=  178 fps=166 q=29.0 size=     512kB time=00:00:06.17 bitrate= 679.3kbits/s speed=5.75x    
frame=  186 fps=138 q=-1.0 Lsize=    1117kB time=00:00:06.17 bitrate=1482.2kbits/s speed=4.57x    
video:1037kB audio:72kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.719310%
[libx264 @ 0x180c20c0] frame I:1     Avg QP:21.37  size: 45488
[libx264 @ 0x180c20c0] frame P:47    Avg QP:21.89  size: 12317
[libx264 @ 0x180c20c0] frame B:138   Avg QP:24.32  size:  3168
[libx264 @ 0x180c20c0] consecutive B-frames:  1.1%  0.0%  0.0% 98.9%
[libx264 @ 0x180c20c0] mb I  I16..4:  8.4% 73.1% 18.6%
[libx264 @ 0x180c20c0] mb P  I16..4:  2.4%  5.8%  0.5%  P16..4: 42.2%  9.4%  6.1%  0.0%  0.0%    skip:33.7%
[libx264 @ 0x180c20c0] mb B  I16..4:  0.4%  0.5%  0.0%  B16..8: 42.1%  1.8%  0.2%  direct: 1.5%  skip:53.4%  L0:45.2% L1:52.8% BI: 1.9%
[libx264 @ 0x180c20c0] 8x8 transform intra:65.1% inter:82.7%
[libx264 @ 0x180c20c0] coded y,uvDC,uvAC intra: 29.1% 32.2% 7.5% inter: 6.0% 9.2% 0.6%
[libx264 @ 0x180c20c0] i16 v,h,dc,p: 58% 14% 10% 18%
[libx264 @ 0x180c20c0] i8 v,h,dc,ddl,ddr,vr,hd,vl,hu: 27% 11% 41%  3%  3%  4%  3%  4%  3%
[libx264 @ 0x180c20c0] i4 v,h,dc,ddl,ddr,vr,hd,vl,hu: 36% 19% 15%  4%  6%  5%  6%  5%  4%
[libx264 @ 0x180c20c0] i8c dc,h,v,p: 61% 14% 21%  3%
[libx264 @ 0x180c20c0] Weighted P-Frames: Y:2.1% UV:2.1%
[libx264 @ 0x180c20c0] ref P L0: 58.8%  8.7% 20.9% 11.6%  0.0%
[libx264 @ 0x180c20c0] ref B L0: 81.9% 13.5%  4.6%
[libx264 @ 0x180c20c0] ref B L1: 92.2%  7.8%
[libx264 @ 0x180c20c0] kb/s:1369.81
