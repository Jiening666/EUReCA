from options.train_options import TrainOptions


def main():
    opt = TrainOptions().parse()
    from train_single import train
    train(opt)


if __name__ == '__main__':
    main()
