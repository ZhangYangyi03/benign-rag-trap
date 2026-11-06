# Changelog

All notable changes are documented here. This file starts with the
2026-11 GitHub portfolio audit: the fixes below were made in response to
`GitHub-项目综合评价.md` and are intentionally itemised so a reviewer can
verify each one against the commit.

## [Unreleased] — 2026-11 portfolio audit fixes

- Added scripts/measurement_validity.py: exact classifier replay (660/660), seven non-degenerate labelling perturbations, bootstrap CIs, and a 60-item human-labelling template.
- Withdrew the unsupported '>95% human agreement' claim and corrected the pattern counts to 27 refusal + 11 advice.
- Scoped the manuscript title, abstract, introduction, limitations and appendix to rule-labeler outputs; added an explicit measurement-validity section.
- Marked the cover letter as a draft that has not been sent; removed the journal-submission claim.
- Added scripts/check_claim_scope.py as a CI guard against future overclaims.

## Audit context

- Full portfolio assessment: `GitHub-项目综合评价.md` (overall 6.8/10; external validation 1.5/10).
- Canonical maintainer identity: `ZhangYangyi03` / Yangyi Zhang.
- This changelog does not claim external validation. See the "Independent reproduction" issue template.
