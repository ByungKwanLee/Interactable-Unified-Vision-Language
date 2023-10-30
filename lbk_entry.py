from utils.arguments import load_opt_command

def main(args=None):
    opt, _ = load_opt_command(args)

    from trainer import XDecoder_Trainer as Trainer
    trainer = Trainer(opt)
    trainer.train()
    trainer.eval()

if __name__ == "__main__":
    main()