import collections

from mmcv.utils import build_from_cfg

from ..builder import PIPELINES


@PIPELINES.register_module()
class Compose(object):
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        print("***" * 20)
        print('before compose:\n', data)
        print("***" * 20)
        k=1
        for t in self.transforms:
            data = t(data)
            if k<7:
                print("***" * 20)
                print(t)
                print('{} th composing {}:\n'.format(k,data['img'].shape))
                print("***" * 20)
            else:
                print("***" * 20)
                print(t)
                print('{} th composing {}:\n'.format(k,(data['img'].data)[0].shape))
                print("***" * 20)
            k=k+1

            if data is None:
                return None
        print("***" * 20)
        print('after compose:\n', (data['img'].data)[0].shape)
        print("***" * 20)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
