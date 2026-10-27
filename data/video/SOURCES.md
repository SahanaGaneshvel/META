# Video sources

All clips are derived from one public-domain source. Attribution isn't required, but we give it anyway.

| File | Derived from | Segment | Processing |
|---|---|---|---|
| `lot_wide_5fps.mp4` (770 frames, 154 s, 1280×720) | Source below | 1–155 s (camera static, wide view, about 30 cars) | 5 fps, H.264 CRF 23, faststart, no audio |
| `lot_close_5fps.mp4` (315 frames, 63 s, 1280×720) | Source below | 253–316 s (camera static, close view of painted bays; one departure in the front row) | same |
| `lot_wide.jpg`, `lot_close.jpg` | Source below | stills at 30 s and 280 s | 1280 px wide |

**Source:** "BLK-HDPTZ12 Security Camera Parking Lot Surveillance Video", by Supercircuits (2012-10-23).
https://commons.wikimedia.org/wiki/File:BLK-HDPTZ12_Security_Camera_Parking_Lot_Surveillance_Video.webm
License: **Public domain** (Wikimedia Commons: `pd`, Copyrighted = False, AttributionRequired = false).
The original file is in `src/` (67 MB); `python scripts/fetch_data.py` re-downloads it and `python scripts/make_video_clips.py` regenerates the clips.

Notes
- Static segments were found by phase-correlation motion analysis. The camera pans and zooms from 155 to 252 s, so that part is excluded.
- 5 fps is deliberate: SAM 3.1 video is billed per frame, and occupancy changes over minutes.
- SAM 3.1 tracks at most 16 objects per frame, so define the "EV bays" as a polygon block of 8–12 bays and crop or mask to it.
- Rejected candidates: a Tower Fall parking timelapse (handheld, never static) and a Mazatlán supermarket clip (shot from a moving vehicle).
