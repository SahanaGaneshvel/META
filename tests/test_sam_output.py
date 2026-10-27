from headroom.meta.sam_output import parse_boxes

SAMPLE = (
    "<0f>0<|box;x1=10;y1=20;x2=110;y2=220;w=1280;h=720;c=0.91|><|mask;x=0;y=0;data=720,1280,~a|b,c|>,"
    "2<|box;x1=300;y1=40;x2=420;y2=260;w=1280;h=720;c=0.88|><|mask;x=0;y=0;data=720,1280,!zz|>\n"
    "<3f>2<|box;x1=305;y1=42;x2=425;y2=262;w=1280;h=720|><|mask;x=0;y=0;data=720,1280,!q|>\n"
)


def test_frames_ids_and_fields():
    frames = parse_boxes(SAMPLE)
    assert sorted(frames) == [0, 3]
    assert [b.object_id for b in frames[0]] == [0, 2]
    assert frames[0][0].confidence == 0.91 and frames[3][0].confidence is None
    assert frames[3][0].x1 == 305 and frames[3][0].centre == (365.0, 152.0)


def test_mask_payload_with_pipes_and_commas_does_not_break_parsing():
    assert len(parse_boxes(SAMPLE)[0]) == 2


def test_empty_output_means_no_detections():
    assert parse_boxes("") == {}
