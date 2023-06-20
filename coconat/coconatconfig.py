COCONAT_ROOT = "/app/coconat"

COCONAT_PLM_DIR = "/mnt/plms"

COCONAT_REGISTER_MODEL = "%s/models/dlModel" % COCONAT_ROOT

COCONAT_REGISTER_MODEL_TORCH = "%s/models/dlModel.ckpt" % COCONAT_ROOT

COCONAT_CRF_MODEL = "%s/models/crfModel" % COCONAT_ROOT

COCONAT_OLIGO_MODEL = "%s/models/oligoModel.ckpt" % COCONAT_ROOT

CRF_BIN = "%s/tools/biocrf-static" % COCONAT_ROOT

PROT_T5_MODEL = "%s/prot_t5_xl_uniref50" % COCONAT_PLM_DIR

ESM_MODEL = "%s/esm2/esm2_t33_650M_UR50D.pt" % COCONAT_PLM_DIR

DEVICE = "cpu"
