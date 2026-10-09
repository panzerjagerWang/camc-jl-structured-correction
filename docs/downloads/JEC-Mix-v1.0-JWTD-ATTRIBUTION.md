# Attribution and changes

**Japanese Wikipedia Typo Dataset (JWTD), version 2.0**, released 16 March 2021
by Yu Tanaka, Yugo Murawaki, Daisuke Kawahara and Sadao Kurohashi,
Kyoto University Language Media Processing Lab. [Official publisher](https://nlp.ist.i.kyoto-u.ac.jp/index.php?%E6%97%A5%E6%9C%AC%E8%AA%9EWikipedia%E5%85%A5%E5%8A%9B%E8%AA%A4%E3%82%8A%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88) ·
[official archive](https://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JWTD/jwtd_v2.0.tar.gz).
The underlying text is by Japanese Wikipedia contributors. No endorsement by
the source authors, Kyoto University or Wikimedia is implied.

Every released row retains the article ID/title, pre/post revision IDs, native
diff/category/likelihoods and original source-line SHA-256 in `publisher_metadata`
and provenance fields. Use its `page`, `pre_rev` and `post_rev` to attribute the
article and contributors:

- Article: `https://ja.wikipedia.org/wiki/?curid=PAGE`
- Revision before: `https://ja.wikipedia.org/w/index.php?oldid=PRE_REV`
- Revision after: `https://ja.wikipedia.org/w/index.php?oldid=POST_REV`
- Contributor history: `https://ja.wikipedia.org/w/index.php?curid=PAGE&action=history`
- Revision difference: `https://ja.wikipedia.org/w/index.php?oldid=PRE_REV&diff=POST_REV`

**Changes by JEC-Mix contributors (2026):** training-only filtering, deterministic
pair selection, overlap-family screening, automatic supplied-span offsets,
insertion/deletion anchors, new provenance hashes and family identifiers.
Source and corrected sentences and all original publisher metadata remain
literal. This is a derived 50,000-row subset, not the complete publisher corpus
and not new human learner annotation.

The derived artifact and its contributed dataset metadata are licensed under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), with the full
legal text in `LICENSE-CC-BY-SA-3.0.txt`. Preserve attribution, identify changes,
link the license, and distribute adaptations under its share-alike terms.
The existing metadata/results data retain CC BY 4.0; code retains MIT; model
weights retain their separate licenses. No blanket license covers the whole mix.

Cite the original dataset authors and the JEC-Mix version when using this subset.
Original English description: [Tanaka et al. (2020), Building a Japanese Typo
Dataset from Wikipedia’s Revision History](https://aclanthology.org/2020.acl-srw.31/).
The publisher's v2 reference is Tanaka, Murawaki, Kawahara and Kurohashi (2021),
日本語Wikipediaの編集履歴に基づく入力誤りデータセットと訂正システムの改良,
言語処理学会第27回年次大会. Machine-readable references are in `citation.bib`.
