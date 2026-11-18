import unittest

import torch

from torch_rating import BerrarAdapter, BerrarLayer, EloAdapter, PiAdapter


class AdapterTests(unittest.TestCase):
    def test_shapes_gradients_and_checkpoint(self):
        for cls in (EloAdapter, BerrarAdapter, PiAdapter):
            for shape in ((4,), (3, 4), (2, 3, 4)):
                with self.subTest(adapter=cls.__name__, shape=shape):
                    model = cls(4, rating_dim=4, out_dim=5).double()
                    home = torch.randn(shape, dtype=torch.float64, requires_grad=True)
                    away = torch.randn(shape, dtype=torch.float64, requires_grad=True)
                    output = model(home, away)
                    self.assertEqual(output.shape, (*shape[:-1], 5))
                    output.square().sum().backward()
                    for value in (home, away):
                        self.assertTrue(torch.isfinite(value.grad).all())
                    restored = cls(4, rating_dim=4, out_dim=5).double()
                    restored.load_state_dict(model.state_dict())
                    torch.testing.assert_close(output, restored(home, away))

    def test_rating_formulas_and_order(self):
        home = torch.tensor([[1., 2., 3., 4.]])
        away = torch.tensor([[5., 6., 7., 8.]])
        elo = EloAdapter(4, rating_dim=4)
        p_home = torch.sigmoid(home - away)
        torch.testing.assert_close(
            elo.rating_features(home, away), torch.cat((1 - p_home, p_home), -1)
        )
        berrar = BerrarAdapter(4, rating_dim=4)
        expected = 80 * torch.sigmoid(torch.cat((away[:, :2] + home[:, 2:], home[:, :2] + away[:, 2:]), -1))
        torch.testing.assert_close(berrar.rating_features(home, away), expected)
        pi = PiAdapter(4, rating_dim=4)
        margin = pi.signed_score(home[:, :2]) - pi.signed_score(away[:, 2:])
        torch.testing.assert_close(pi.rating_features(home, away), torch.cat((-margin, margin), -1))
        zero = torch.zeros(2, requires_grad=True)
        pi.signed_score(zero).sum().backward()
        self.assertTrue((zero.grad > 0).all())

    def test_freezing_and_validation(self):
        for cls in (EloAdapter, BerrarAdapter, PiAdapter):
            model = cls(4, learnable_parameters=False)
            for name, parameter in model.named_parameters():
                self.assertEqual(parameter.requires_grad, name.startswith(('project.', 'encode.')))
            with self.assertRaises(ValueError):
                model(torch.zeros(2, 4), torch.zeros(3, 4))
            with self.assertRaises(ValueError):
                model(torch.zeros(2, 3), torch.zeros(2, 3))
        for cls in (BerrarAdapter, PiAdapter):
            with self.assertRaises(ValueError):
                cls(4, rating_dim=3)

    def test_legacy_berrar_uses_away_defense(self):
        layer = BerrarLayer(2)
        home = torch.tensor([0., 0.])
        first = layer(home, torch.tensor([0., 0.]))
        second = layer(home, torch.tensor([0., 1.]))
        torch.testing.assert_close(first[0], second[0])
        self.assertGreater(second[1], first[1])


if __name__ == '__main__':
    unittest.main()
