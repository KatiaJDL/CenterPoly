from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .exdet import ExdetDetector
from .ddd import DddDetector
from .ctdet import CtdetDetector
from .polydet import PolydetDetector
from .diskdet import DiskdetDetector
from .gaussiandet import GaussianDetector
from .multi_pose import MultiPoseDetector

detector_factory = {
  'exdet': ExdetDetector, 
  'ddd': DddDetector,
  'ctdet': CtdetDetector,
  'polydet': PolydetDetector,
  'diskdet': DiskdetDetector,
  'gaussiandet': GaussianDetector,
  'multi_pose': MultiPoseDetector,
}
