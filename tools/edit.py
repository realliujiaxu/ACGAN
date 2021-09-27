import fire
from acgan_config import face_args
from stylegan2_pytorch import Trainer
from stylegan2_pytorch.helper import add_attr


def edit(img_path='./scene256.jpg'):
    model_args = face_args

    model = Trainer(**model_args)
    model.load(model_args.load_from)

    # TODO: Edit this function as you need
    def attribute_processor(attr):
        attr_list = []
        attr_list.append(attr)
        add_attr(attr_list, [31], attr, 0.4)
        add_attr(attr_list, [-8, -11, 9], attr, 0.4)
        add_attr(attr_list, [-8, 17, -39], attr, 0.3)
        add_attr(attr_list, [9, 31], attr, 0.4)
        return attr_list

    save_path = model.edit(img_path, attribute_processor)
    print(f'edited real images generated at {save_path}')
    return


def main():
    fire.Fire(edit)
