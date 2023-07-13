import wandb


class WandbLogger:
    def __init__(self, configs):
        wandb.init(
            name=configs["wandb"]["name"],
            dir=configs["wandb"]["dir"],
            project=configs["wandb"]["project"],
            resume="allow",
            entity=configs["wandb"]["entity"],
            tags=configs["wandb"]["tags"],
            config=configs
        )

    @property
    def dir(self):
        return wandb.run.dir

    def log_image(self, key, image, step):
        wandb_image = wandb.Image(image)
        self.log(log_dict={key: wandb_image}, step=step)

    def log_obj(self, key, obj_file, step):
        wandb.log({key: [wandb.Object3D(open(obj_file))]}, step=step)

    def log(self, log_dict, step):
        wandb.log(log_dict, step=step)
