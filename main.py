import argparser
from main_launcher.plot_history_launcher import decode_plot_args, launch_plot
from main_launcher.train_launcher import decode_trainer_args, launch_train

if __name__ == '__main__':

    args = argparser.args_parser()
    print("args : ", args)

    actions = {
        "train": (decode_trainer_args, launch_train),
        "plot": (decode_plot_args, launch_plot)
    }

    action = args["action"]

    try:
        decoder, launcher = actions[action]
    except KeyError:
        raise NotImplementedError(f"Action : '{action}' is not handled.")

    kwargs = decoder(args)
    launcher(**kwargs)
