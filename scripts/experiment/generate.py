import argparse
import copy
import csv
import os
import yaml
from datetime import datetime
from itertools import product


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))


cwd = os.getcwd()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", default="./exp_confs/pw_2_stilts.yaml", type=str)
    parser.add_argument("--vary_path", default="./exp_confs/trng_parms_to_vary.yaml", type=str)
    parser.add_argument("--remove_base_conf", action="store_true")
    args = parser.parse_args()

    base_exp_conf_file = open(args.base_path)
    base_exp_conf = yaml.load(base_exp_conf_file, Loader=yaml.FullLoader)

    # ./logs/<task_name>/<experiment_name>/<variant_name>/
    task_name = base_exp_conf["env_name"].split("-")[:-1][0].lower()
    dtn = datetime.now().strftime("%Y%b%d_%H-%M-%S")
    experiment_name = dtn + "_" + base_exp_conf["run_name"]

    vary_trng_params_file = open(args.vary_path)
    params_to_vary = yaml.load(vary_trng_params_file, Loader=yaml.FullLoader)

    variant_list = list(product_dict(**params_to_vary))

    print("n. variants:", len(variant_list))

    exp_id = 0

    logdir_exp = cwd + "/logs/" + task_name + "/" + experiment_name + "/"
    os.makedirs(logdir_exp, exist_ok=True)
    # dump the base_expe_config and vary_trng_params
    dumpfile_at = logdir_exp + "base_exp_conf.yaml"
    base_exp_conf_file = open(dumpfile_at, "w")
    yaml.dump(base_exp_conf, base_exp_conf_file, default_flow_style=False, sort_keys=False)
    dumpfile_at = logdir_exp + "vary_exp_params.yaml"
    vary_trng_params_file = open(dumpfile_at, "w")
    yaml.dump(params_to_vary, vary_trng_params_file, default_flow_style=False, sort_keys=False)

    with open(logdir_exp + "/param_vary_list.csv", "w") as pvl:
        csv_writer = csv.writer(pvl)

        for variant_id, variant in enumerate(variant_list):
            # merge all the values of variant, with a '_' separating each , relace "." with "p"

            for v in variant.values():
                if not isinstance(v, int) and not isinstance(v, float) and not isinstance(v, str):
                    print(
                        "note:variant value not a int, float or str to reflect in folder name, defaulting to variant id"
                    )
                    variant_name = "_".join(["var_" + str(variant_id)])
                else:
                    variant_name = "_".join([str(v) for v in variant.values()]).replace(".", "p")

            this_variant = copy.deepcopy(base_exp_conf)

            if variant_id == 0:
                header_names = ["exp_id"] + [pn for pn in variant.keys()]
                print(header_names)
                csv_writer.writerow(header_names)

            csv_writer.writerow([str(exp_id)] + [pv for pv in variant.values()])

            for param_name in variant.keys():
                val = this_variant
                param_path = param_name.split("/")
                param_val = variant[param_name]

                for i in range(len(param_path) - 1):
                    # print(val.keys(),param_path[i])
                    val = val[param_path[i]]
                # print(val)
                # if param_path[-1] in val.keys():
                #     val.pop(param_path[-1])
                val.update({param_path[-1]: param_val})

            logdir_exp_variant = logdir_exp + variant_name + "/"
            os.makedirs(logdir_exp_variant, exist_ok=True)

            this_variant.update({"run_name": experiment_name + "/" + variant_name})
            dumpfile_at = logdir_exp_variant + "exp_conf.yaml"

            trng_exp_conf_file = open(dumpfile_at, "w")
            yaml.dump(this_variant, trng_exp_conf_file, default_flow_style=False, sort_keys=False)
            print("dumped file:", dumpfile_at)
            exp_id += 1

        if args.remove_base_conf:
            os.remove(args.base_path)
            print("deleted file:", args.base_path)
