from tbma_gnas.search_space.block import LearnableBlock


class TestBlock:

    def test_output_block(self):
        block = LearnableBlock(is_output=True)
        layer, _ = block.query(prev_out_shape=15, num_node_features=25, data_out_shape=6)

        assert layer.out_channels == 6

    def test_learning(self):
        block = LearnableBlock()
        layer, act = block.query(prev_out_shape=10, num_node_features=25, data_out_shape=5)

        block.learn(layer=layer, activation=act, positive=True, reg=None)
        assert any(score > 1 for score in block.layer.get_scores().values())
