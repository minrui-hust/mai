
class BaseCodec(object):
    r'''
    codec do all the things that is task dependent, which are:
        1. encode 'data' and 'anno' into task specific 'input' and 'gt',
        2. decode task specific 'output' into 'anno'
        3. calc loss given output and gt
        5. define the collator to collate the encoded sample
        4. and also a plot method to viz encoded sample
    codec has four mode, which is:
        1. train, given output, calc loss
        2. eval, decode 'output' to 'anno' for evaluation
        3. export, decode 'output' for onnx export
        4. trt, decode the output of tensorrt model to 'anno' for evaluation, this used to evaluate the tensorrt model performance
    '''

    def __init__(self,
                 encode_cfg={'encode_data': True, 'encode_anno': True},
                 decode_cfg={},
                 loss_cfg={},
                 ):
        super().__init__()
        self.mode = 'train'
        self.encode_cfg = encode_cfg
        self.decode_cfg = decode_cfg
        self.loss_cfg = loss_cfg

    def set_train(self):
        self.set_mode('train')

    def set_eval(self):
        self.set_mode('eval')

    def set_export(self):
        self.set_mode('export')

    def set_trt(self):
        self.set_mode('trt')

    def set_mode(self, mode):
        assert(mode in ['train', 'eval', 'export', 'trt'])
        self.mode = mode

    def encode(self, sample, info):
        r'''
        data --> input
        anno --> gt
        after encode, sample would be like this:
        {'input', 'gt', 'data', 'anno', 'meta'}
        '''
        if self.encode_cfg['encode_data']:
            self.encode_data(sample, info)
        if self.encode_cfg['encode_anno']:
            self.encode_anno(sample, info)

    def decode(self, output, batch=None):
        r'''
        output --> pred
        '''
        if self.mode == 'train':
            return self.decode_train(output, batch)
        elif self.mode == 'eval':
            return self.decode_eval(output, batch)
        elif self.mode == 'export':
            return self.decode_export(output, batch)
        elif self.mode == 'trt':
            return self.decode_trt(output, batch)
        else:
            raise NotImplementedError

    def decode_train(self, output, batch):
        return self.loss(output, batch)

    def decode_eval(self, output, batch):
        raise NotImplementedError

    def decode_export(self, output, batch):
        raise NotImplementedError

    def decode_trt(self, output, batch):
        raise NotImplementedError

    def encode_data(self, sample, info):
        raise NotImplementedError

    def encode_anno(self, sample, info):
        raise NotImplementedError

    def loss(self, output, batch):
        raise NotImplementedError

    def get_collater(self):
        raise NotImplementedError

    def plot(self, sample, **kwargs):
        raise NotImplementedError
