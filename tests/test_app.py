from tests.client import client


urls = {
    'bad_url': 'https://vk.com',
    'not_found': 'https://vk.com/not_exists.jpeg',
    'correct': 'https://cdn.vox-cdn.com/thumbor/6ofCV2k-SeKBJytacICi1mrzvL8=/0x66:1600x1133/1200x800/filters:focal(0x66:1600x1133)/cdn.vox-cdn.com/uploads/chorus_image/image/37575328/hello_maybe_not_kitty.0.0.jpeg',
}


class TestApp:
    @staticmethod
    def is_response_image_404(image):
        error = image.get('err')
        assert error.get('code') == 404

    @staticmethod
    def is_response_image_400(image):
        error = image.get('err')
        assert error.get('code') == 400

    @staticmethod
    def is_response_image_predicted(image):
        prediction = image.get('prediction')
        assert prediction
        assert len(prediction) > 0

    @staticmethod
    def check_response(response):
        assert response.status_code == 200
        assert response.headers.get('content-type') == 'application/json'
        assert response.json().get('images')
        assert len(response.json().get('images')) == len(urls)

    def test_predict(self):
        response = client.post(
            url='/predict',
            json={
                'model': 'vit',
                'images': [
                    {
                        'url': urls.get('bad_url'),
                    },
                    {
                        'url': urls.get('not_found'),
                    },
                    {
                        'url': urls.get('correct'),
                    },
                ]
            },
        )

        self.check_response(response)

        for image in response.json().get('images'):
            url = image.get('url')

            if url == urls.get('bad_url'):
                self.is_response_image_400(image)

            if url == urls.get('not_found'):
                self.is_response_image_404(image)

            if url == urls.get('correct'):
                self.is_response_image_predicted(image)
