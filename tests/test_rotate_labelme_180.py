from scripts.rotate_labelme_180 import rotate_labelme_document_180


def test_rotate_labelme_document_180_uses_pixel_center_coordinates() -> None:
    document = {
        "imageWidth": 1920,
        "imageHeight": 1080,
        "shapes": [
            {"label": "line", "points": [[0.0, 0.0], [1919.0, 1079.0]]},
            {"label": "pocket", "points": [[320.5, 240.25]]},
        ],
    }

    rotated = rotate_labelme_document_180(document)

    assert rotated["shapes"][0]["points"] == [[1919.0, 1079.0], [0.0, 0.0]]
    assert rotated["shapes"][1]["points"] == [[1598.5, 838.75]]
    assert document["shapes"][0]["points"][0] == [0.0, 0.0]


def test_rotate_labelme_document_180_is_an_involution() -> None:
    document = {
        "imageWidth": 4,
        "imageHeight": 3,
        "shapes": [{"label": "outline", "points": [[0.25, 1.5], [3.0, 2.0]]}],
    }

    assert rotate_labelme_document_180(rotate_labelme_document_180(document)) == document
