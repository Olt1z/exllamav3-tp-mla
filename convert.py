from exllamav3.conversion.convert_model import parser, main, prepare

# Script included in package: ./exllamav3/conversion/convert_model.py

if __name__ == "__main__":
    _args = parser.parse_args()
    _in_args, _job_state, _ok, _err = prepare(_args)
    if not _ok:
        # rc 1: a receita ou os argumentos recusados têm de derrubar o job que chamou (pipefail)
        print(f" !! Error: {_err}", flush = True)
        raise SystemExit(1)
    else:
        main(_in_args, _job_state)