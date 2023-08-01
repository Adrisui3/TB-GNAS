from torch_geometric import nn as geom_nn

from tbma_gnas.search_space.component_type import ComponentType
from tbma_gnas.search_space.space_component import LearnableSpaceComponent


class TestSpaceComponent:

    def test_non_negative_scores(self):
        layer = LearnableSpaceComponent(component_type=ComponentType.LAYER)
        for sc in layer.get_scores().values():
            assert sc == 1

    def test_learning_component(self):
        layer = LearnableSpaceComponent(component_type=ComponentType.LAYER)

        layer.learn(geom_nn.GATConv.__name__, True)
        assert layer.get_scores()[geom_nn.GATConv.__name__] == 2

        layer.learn(geom_nn.GATConv.__name__, False)
        assert layer.get_scores()[geom_nn.GATConv.__name__] == 1
