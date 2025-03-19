import numpy as np

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, dataset, input_dimension=6, output_dimension=6, width=300):

        if "Sin" == dataset:

            if model_type == "representation":

                hidden_size = width
                return [

                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": input_dimension}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": True, "meta": True,
                     "config": {"out": output_dimension, "in": hidden_size}}
                ]

        # elif dataset == "omniglot":
        #     channels = 256
        #
        #     return [
        #         {"name": 'conv2d', "adaptation": False, "meta": True,
        #          "config": {"out-channels": channels, "in-channels": 1, "kernal": 3, "stride": 2, "padding": 0}},
        #         {"name": 'relu'},
        #
        #         {"name": 'conv2d', "adaptation": False, "meta": True,
        #          "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 1,
        #                     "padding": 0}},
        #         {"name": 'relu'},
        #
        #         {"name": 'conv2d', "adaptation": False, "meta": True,
        #          "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
        #                     "padding": 0}},
        #         {"name": 'relu'},
        #         #
        #         {"name": 'conv2d', "adaptation": False, "meta": True,
        #          "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 1,
        #                     "padding": 0}},
        #         {"name": 'relu'},
        #
        #         {"name": 'conv2d', "adaptation": False, "meta": True,
        #          "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
        #                     "padding": 0}},
        #         {"name": 'relu'},
        #
        #         {"name": 'conv2d', "adaptation": False, "meta": True,
        #          "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
        #                     "padding": 0}},
        #         {"name": 'relu'},
        #
        #         {"name": 'flatten'},
        #         # {"name": 'rotate'},
        #         {"name": 'rep'},
        #
        #         {"name": 'linear', "adaptation": True, "meta": True,
        #          "config": {"out": 1000, "in": 9 * channels}}
        #
        #     ]

        elif dataset == "omniglot":
            channels = 256

            return [
                # RLN
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": 1, "kernal": 3, "stride": 2, "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},
                #
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'flatten'},
                # {"name": 'rotate'},


                {"name": 'linear', "adaptation": False, "meta": True,
                 "config": {"out": 1000, "in": 9 * channels}},

                {"name": 'rep'},

                # PLN
                {"name": 'relu'},

                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": 1000, "in": 1000}}

            ]
        elif dataset == 'mnist':
            channels = 64
            return [
                # RLN
                # [b,c,28,28]
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": 3, "kernal": 3, "stride": 2, "padding": 0}},
                {"name": 'relu'},
                # [b,c,13,13]
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},
                # [b,c,6,6]
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 2, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},
                # [b,c,2,2]

                {"name": 'flatten'},
                # {"name": 'rotate'},
                # [b,c,6,6]

                {"name": 'rep'},

                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": 10, "in": 3 * 3 * channels}},


            ]


        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)
