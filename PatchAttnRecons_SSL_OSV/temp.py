from model import *

def main():
    args = {'optimizer':'SGD','is_pretrained':False,'learning_rate':0.1,'warmup_epochs':10,'max_epochs':200}
    ssl_model = SSL_Model(args)    
    if args.load_model is not None:
        checkpoint = torch.load(".\saved_models\BHSig260_Hindi_R=2_SSL.pt")
        ssl_model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epochs']
        torch.save(ssl_model.encoder.state_dict(), "./saved_models/BHSig260_Hindi_R=2_SSL_Encoder_RN18_190ep.pth")
        print(f">> Resume training from {epoch} epochs")

if __name__ == '__main__':
    main()
