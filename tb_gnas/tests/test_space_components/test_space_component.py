from torch_geometric import nn as geom_nn

from tb_gnas.search_space.component_type import ComponentType
from tb_gnas.search_space.learnable_space_component import LearnableSpaceComponent


class TestSpaceComponent:

    def test_non_negative_scores(self):
        layer = LearnableSpaceComponent(component_type=ComponentType.ATTENTION)
        for sc in layer.get_scores().values():
            assert sc == 1

    def test_learning_component(self):
        layer = LearnableSpaceComponent(component_type=ComponentType.ATTENTION)

        layer.learn("gat", True)
        assert layer.get_scores()["gat"] == 2

        layer.learn("gat", False)
        assert layer.get_scores()["gat"] == 1
