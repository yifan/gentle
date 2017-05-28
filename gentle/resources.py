import logging
import os

from util.paths import get_resource, ENV_VAR
from gentle import metasentence

class Resources:

    def __init__(self, modelDir):
        self.proto_langdir = get_resource(modelDir)
        self.nnet_gpu_path = get_resource(os.path.join(modelDir, 'online'))
        self.full_hclg_path = get_resource(os.path.join(self.nnet_gpu_path, 'graph', 'HCLG.fst'))

        def require_dir(path):
            if not os.path.isdir(path):
                raise RuntimeError("No resource directory %s.  Check %s environment variable?" % (path, ENV_VAR))


        require_dir(self.proto_langdir)
        require_dir(self.nnet_gpu_path)

        with open(os.path.join(self.proto_langdir, "langdir", "words.txt")) as fh:
            self.vocab = metasentence.load_vocabulary(fh)


