# AudioMuse-AI-SAE

A sparse autoencoder over the [DCLAP](https://github.com/NeptuneHub/AudioMuse-AI-DCLAP)
music embedding space, and the concept vocabulary it discovers.

## What it is

DCLAP turns a song into 512 numbers. Those numbers are dense and unnamed: every
track uses all of them and no single one means anything you can point at.

The SAE re-describes the same song as a handful of entries from a dictionary of
**1024 latents**, of which only about twenty fire per track. Training is
unsupervised, so the latents are discovered rather than labelled; a separate
offline pass then maps text terms onto them, producing a catalogue of concepts
like `saxophone`, `female vocals`, `heavy metal` or `viola`.

That makes a search refinable. You can ask for more of a concept, or less of it,
and rank results by how much of it each track actually contains.

Three artifacts come out of training:

| File | Size | Role |
| --- | --- | --- |
| `dclap_sae_k20_d1024_best_encoder.onnx` | 2 MB | embedding -> 1024 latent activations |
| `dclap_sae_k20_d1024_best_decoder.onnx` | 2 MB | latent activations -> embedding |
| `dclap_sae_concepts.json` | 75 KB | which latents each term maps to |

## Quick start

Everything below is a copy-paste. It ends with a real embedding printed to your
terminal.

**1. Get the models.** Two releases, one folder:

```bash
mkdir -p models && cd models

# SAE: the dictionary and its concept catalogue
curl -LO https://github.com/NeptuneHub/AudioMuse-AI-SAE/releases/download/v1/dclap_sae_k20_d1024_best_encoder.onnx
curl -LO https://github.com/NeptuneHub/AudioMuse-AI-SAE/releases/download/v1/dclap_sae_k20_d1024_best_decoder.onnx
curl -LO https://github.com/NeptuneHub/AudioMuse-AI-SAE/releases/download/v1/dclap_sae_concepts.json

# DCLAP: the text tower, to turn a query into an embedding
curl -LO https://github.com/NeptuneHub/AudioMuse-AI-DCLAP/releases/download/v1/clap_text_model.onnx
cd ..
```

Add these two if you also want to embed audio yourself:

```bash
curl -L -o models/model_epoch_36.onnx https://github.com/NeptuneHub/AudioMuse-AI-DCLAP/releases/download/v1/model_epoch_36.onnx
curl -L -o models/model_epoch_36.onnx.data https://github.com/NeptuneHub/AudioMuse-AI-DCLAP/releases/download/v1/model_epoch_36.onnx.data
```

**2. Install the dependencies.** Three packages, no GPU needed:

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy onnxruntime transformers
```

**3. Run it:**

```bash
python example.py
```

```
query   : "POP Viola with Female vocalist"
enforce : more viola at strength 1.0

ORIGINAL query embedding
  [+0.0547 -0.0451 -0.0618 +0.0694 +0.0405 -0.0266 +0.0110 +0.0300 ...]  (512 values, norm 1.0000)

NEW query embedding after enforcing the concept
  [+0.0395 -0.0595 -0.0670 +0.0628 +0.0267 -0.0225 +0.0088 +0.0374 ...]  (512 values, norm 1.0000)

concept "viola" uses 12 of 1024 latents, 12 of them were changed
cosine(original, new) = 0.9691   (1.0 would mean no change)

Search your index with the NEW embedding to get the refined results.
```

The new vector is a drop-in replacement for the original: search your DCLAP
index with it and the results lean toward viola.

Other concepts, strengths and directions:

```bash
python example.py --concept "female vocals" --strength 3
python example.py --concept techno --direction less
python example.py --query "calm piano at night" --concept piano --strength 10
```

`--strength` takes 1, 3, 5 or 10. Each concept is a unit norm mask over its
latents, so the same number means the same size of step for every concept. Low
values nudge the query, high values start to overwrite it. The paper's own grid
stops at 2.0, which on this dictionary is close to a no-op. See
[`example.py`](example.py) for the 40 lines that do the work.

## How the edit works

The query embedding is encoded into 1024 sparse concept latents. The requested
concept has a support, a small set of those latents, and a unit norm mask over
it. Amplifying adds `alpha * mask` to those coordinates, suppressing subtracts it
and clamps at zero, since latent activations are non negative. The edited code is
decoded back to 512 dimensions, and the difference between the edited and the
unedited reconstruction is added to the original query:

```
s      = encoder(q)
s'     = s + alpha * mask          (clamped at zero when suppressing)
q'     = q + decoder(s') - decoder(s)
```

Because the mask is unit norm, one strength setting means the same size of step
for every concept in the catalogue. Because only the difference is applied, a
strength of zero leaves the query exactly unchanged.

Notes:

* Your index is never rebuilt or re-quantised. Only the query vector moves.
* Concept quality varies. Every entry carries a `grounding` score: the overlap
  between the tracks its latents fire on and the tracks DCLAP itself ranks
  highest for the same words. Instruments and vocals score well; broad genre
  words such as `rock` or `pop` are diffuse and score near chance.
* Instrument concepts are timbral, not literal. `viola`, `violin` and `cello`
  share latents and behave as a single bowed strings detector.

## Training

The code is in [`SAE/`](SAE) and needs a GPU. It reads one Postgres table, and
only two columns of it, so no titles or artists ever leave the database:

```sql
CREATE TABLE clap_embedding (
    item_id   TEXT PRIMARY KEY,   -- any unique id
    embedding BYTEA               -- raw bytes of a float32 vector, 512 values
);
```

```bash
cd SAE
bash setup_venv.sh --with-optional          # creates SAE/.venv via uv
export DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DBNAME
bash run_all.sh                             # extract -> train -> evaluate -> export
```

`run_all.sh` writes the three artifacts listed above. Pass `--table`,
`--id-column`, `--embedding-column` and `--embedding-dim` to
`extract_embeddings.py` if your schema differs from the defaults.

Naming the latents is a second, separate pass: `validate_concepts.py` scores a
candidate term by checking whether the tracks its latents fire on agree with the
tracks DCLAP itself ranks highest for the same words, keeping only those that do.
`build_concept_bundle.py` then packs the survivors into `dclap_sae_concepts.json`.

## Acknowledgement

The concept discovery and steering method is from:

> Julien Guinot, Alain Riou, Elio Quinton, Gyorgy Fazekas.
> *Steering dense music retrieval with open-vocabulary concept discovery.*
> <https://arxiv.org/abs/2608.08757>, used under
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

This implementation follows the paper's sparse autoencoder, its inversion of a
text concept onto latents (Eq. 4) and its IDF penalty on frequent neurons
(Eq. 5). It departs from the paper in applying concepts as a post-retrieval
conjunction rather than as an edit of the query vector.

## License

AGPL-3.0-only. See [LICENSE](LICENSE).
