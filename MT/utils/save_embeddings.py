import torch

def save_embedding(datasets, encoder, hparams, save_file_dict):

        test_asv_emb_dict, test_cm_emb_dict, test_sasv_scores_dict = encoder.evaluate(
                test_set=datasets["eval"],
                min_key="eer",
                progressbar= True,
                test_loader_kwargs=hparams["validloader_options"],
        )
        torch.save(test_sasv_scores_dict, save_file_dict["test_sasv_scores_file"])
        torch.save(test_asv_emb_dict, save_file_dict["test_asv_emb_file"])
        torch.save(test_cm_emb_dict, save_file_dict["test_cm_emb_file"])