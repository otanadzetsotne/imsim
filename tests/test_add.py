from tests.client import client


def test_add():
    url = '/add'
    body = {
        'model': 'vit',
        'images': [
            {
                'url': 'https://vk.com',
                'id_custom': 56,
            },
            {
                'url': 'https://cdn.vox-cdn.com/thumbor/6ofCV2k-SeKBJytacICi1mrzvL8=/0x66:1600x1133/1200x800/filters:focal(0x66:1600x1133)/cdn.vox-cdn.com/uploads/chorus_image/image/37575328/hello_maybe_not_kitty.0.0.jpeg',
            },
        ]
    }

    response = client.post(
        url=url,
        json=body,
    )

    assert response.status_code == 200
