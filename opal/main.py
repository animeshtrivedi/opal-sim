# SPDX-License-Identifier: Apache-2.0
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opal.util import get_bool_env_var
from opal.opal_profile import profile_function
from opal.opal import OpalSimulator

if __name__ == "__main__":
    print(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    print(sys.path)
    opal = OpalSimulator()
    opal.init_from_cmd_args()

    if True == get_bool_env_var("OPAL_PROFILE", False):
        runtime, virtual_time = profile_function(opal.run)
    else:
        runtime, virtual_time = opal.run()

    del opal  # cleanup
