import argparse
import os
import annotate_fisher

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help='path to extracted LDC2004T19 & LDC2005T19', required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--df-tag', type=bool, help='whether provide disfluency tags or not', default=True)
    args = parser.parse_args()

    labels = annotate_fisher.Annotator(
        Input_path=args.input_path,
        Output_path=args.output_path,
        Model_path=args.model_path,
        Df_tag=args.df_tag
    )
    labels.setup()

if __name__ == "__main__":
    main()