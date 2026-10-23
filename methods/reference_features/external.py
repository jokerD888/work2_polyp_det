from methods.support_reference.external import run

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--job', required=True)
    run(p.parse_args().job)
