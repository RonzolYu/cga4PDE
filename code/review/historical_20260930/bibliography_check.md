> Historical audit of the pre-2026-10-01 package. Its counts, paths, and conclusions are not current revision evidence.

# Bibliography metadata check for the SIAM JSC package

**Scope.** This is a metadata audit of the 20 entries cited by the current `main.aux` in `paper/sisc_cga`.  I checked title, author order/spelling, venue, year, volume/issue, pages or article number, and DOI against Crossref records and publisher/official records where available.  This audit does not assess whether a citation is mathematically sufficient for a particular claim.

## Result

No DOI, author-order, journal, year, volume/issue, or page-range error was found in the cited records.  The entries for EYu2018, LiSiegelEntropy2024, XuXu2025, and MaoSiegelXu2026 are confirmed.  In particular, `EYu2018` correctly renders the surname `E` followed by the given name Weinan; it must not be reordered as “Yu, Bing, and E, Weinan.”

The one metadata discrepancy is the title of `TemlyakovCGA2013`.  The arXiv record and the first page of arXiv:1312.1244 spell the title **“Chebushev Greedy Algorithm in convex optimization”** (with *Chebushev*).  The current BibTeX entry uses the mathematically standard spelling **“Chebyshev Greedy Algorithm in Convex Optimization.”** This is a source-title normalization rather than an author/DOI error.  Choose one policy before submission:

- preserve the arXiv title exactly (`Chebushev ... in convex optimization`) and add `url = {https://arxiv.org/abs/1312.1244}`; or
- retain the corrected standard spelling, but cite it explicitly as the arXiv preprint and recognize that the displayed title is editorially normalized.

The body of the manuscript may continue to use **Chebyshev greedy algorithm** for the algorithm name; that terminology is independent of the typo in the arXiv title.

`XuXu2025` is a journal article with article number 26 in *Journal of Scientific Computing*, volume 105(1), 2025, DOI `10.1007/s10915-025-03050-5`.  The current `pages = {26}` is understandable in the SIAM bibliography output, but `article-number = {26}` (or a note “Article 26”) is more precise if the bibliography style supports it; there is no missing page range to repair.

`MaoXuXu2026` is an arXiv-only preprint (arXiv:2601.11771, submitted January 16, 2026).  Its title and author order are correct.  It should not be presented as a journal publication or assigned volume/pages.

## Entry-by-entry verification

| Key | Verification | Action |
|---|---|---|
| `TemlyakovCGA2013` | Vladimir N. Temlyakov; arXiv:1312.1244; title source spells *Chebushev* | Decide exact-title policy; add arXiv URL recommended |
| `TemlyakovGreedy2014` | *Proceedings of the Steklov Institute of Mathematics* 284(1), 244–262 (2014), DOI 10.1134/S0081543814010180 | No correction |
| `DereventsovTemlyakov2022` | *Applied and Computational Harmonic Analysis* 60, 489–511 (2022), DOI 10.1016/j.acha.2022.05.001 | No correction |
| `SiegelXuOGA2022` | *IEEE Transactions on Information Theory* 68(5), 3354–3361 (2022), DOI 10.1109/TIT.2022.3147984 | No correction |
| `LiSiegelEntropy2024` | *Mathematical Models and Methods in Applied Sciences* 34(5), 779–802 (2024), DOI 10.1142/S0218202524500143 | Confirmed |
| `SiegelHongJinHaoXu2023` | *Journal of Computational Physics* 484, article 112084 (2023), DOI 10.1016/j.jcp.2023.112084 | No correction |
| `SiegelXuVariation2023` | *Constructive Approximation* 57(3), 1109–1132 (2023), DOI 10.1007/s00365-023-09626-4 | No correction |
| `SiegelXuSharp2024` | *Foundations of Computational Mathematics* 24, 481–537 (2024), DOI 10.1007/s10208-022-09595-3 | No correction |
| `XuXu2025` | Jinchao Xu and Xiaofeng Xu; *Journal of Scientific Computing* 105(1), Article 26 (2025), DOI 10.1007/s10915-025-03050-5 | Optional `article-number` field |
| `MaoSiegelXu2026` | *SIAM Journal on Mathematical Analysis* 58(2), 1171–1186 (2026), DOI 10.1137/24M1686693 | Confirmed |
| `BernaFalco2025` | Pablo M. Berná and Antonio Falcó; *Journal of Nonlinear and Variational Analysis* 9(2), 161–177 (2025), DOI 10.23952/jnva.9.2025.2.01 | No correction |
| `BarrettLiu1993` | *Mathematics of Computation* 61(204), 523–537 (1993), DOI 10.1090/S0025-5718-1993-1192966-4 | No correction |
| `LeePark2025` | *IMA Journal of Numerical Analysis* 45(5), 2655–2684 (2025), DOI 10.1093/imanum/drae068 | No correction |
| `DieningKreuzer2008` | *SIAM Journal on Numerical Analysis* 46(2), 614–638 (2008), DOI 10.1137/070681508 | No correction |
| `BrennerScott2008` | 3rd ed., Springer, 2008, DOI 10.1007/978-0-387-75934-0 | No correction |
| `RahimiRecht2007` | NeurIPS 20 (2007), 1177–1184; official NeurIPS record confirms title and authors | No correction |
| `EYu2018` | *Communications in Mathematics and Statistics* 6(1), 1–12 (2018), DOI 10.1007/s40304-018-0127-z | Confirmed; surname `E` is correct |
| `ChenChiEYang2022` | *Journal of Machine Learning* 1(3), 268–298 (2022), DOI 10.4208/jml.220726 | No correction |
| `MaoXuXu2026` | Tong Mao, Jinchao Xu, Xiaofeng Xu; arXiv:2601.11771 (2026) | Keep as preprint only |
| `BarronCohenDahmenDeVore2008` | *The Annals of Statistics* 36(1), 64–94 (2008), DOI 10.1214/009053607000000631 | No correction |

## Sources used

- [arXiv:1312.1244](https://arxiv.org/abs/1312.1244) (Temlyakov title and author).
- [arXiv:2601.11771](https://arxiv.org/abs/2601.11771) (Mao–Xu–Xu title, authors, and date).
- [Crossref](https://api.crossref.org/works/10.1007/s10915-025-03050-5) and [publisher record](https://doi.org/10.1007/s10915-025-03050-5) (Xu–Xu article number).
- [SIAM publisher record](https://epubs.siam.org/doi/10.1137/24M1686693) (Mao–Siegel–Xu volume, pages, DOI).
- [NeurIPS 2007 record](https://proceedings.neurips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html) (Rahimi–Recht).
- [Crossref metadata](https://api.crossref.org/works/10.1142/S0218202524500143) (Li–Siegel).
