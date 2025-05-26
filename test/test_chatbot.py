import unittest
from inference import load_models, generate_response
from app import app

class ChatbotTestCase(unittest.TestCase):
    def setUp(self):
        self.tokenizer, self.enc_model, self.dec_model = load_models()
        app.testing = True
        self.client = app.test_client()

    def test_model_load(self):
        self.assertIsNotNone(self.tokenizer)
        self.assertIsNotNone(self.enc_model)
        self.assertIsNotNone(self.dec_model)

    def test_response_generation(self):
        resp = generate_response('Hola', [], self.tokenizer, self.enc_model, self.dec_model)
        self.assertIsInstance(resp, str)
        self.assertTrue(len(resp) > 0)

    def test_flask_endpoint(self):
        r = self.client.get('/')
        self.assertEqual(r.status_code, 200)
        r2 = self.client.post('/', data={'message': 'Hola'})
        self.assertEqual(r2.status_code, 200)
        self.assertIn(b'Bot:', r2.data)

if __name__ == '__main__':
    unittest.main()
