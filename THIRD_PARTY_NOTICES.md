# Third-party sources

- AgentDojo: https://github.com/ethz-spylab/agentdojo , MIT. Frozen local source distribution0.1.35, benchmarkv1.2.2; license in vendor/agentdojo/LICENSE.
- IPIGuard: https://github.com/Greysahy/ipiguard , Apache-2.0, official commit4e686ed2f62c135cb12564a4466daa95f3ace878. License in reference/IPIGUARD-LICENSE; pinned reference files and their source SHA in reference/reference/ipiguard-provenance.json.
- Task Shield prompts and project adaptation: bundled task_shield_defense.py retains its source header; see docs/METHODS.md for reproduction boundary.
- Semantic Action Frame Guard core and method adapters: included as source dependencies of Holistic, AS-only and Witness-form methods. AS-all and AS-core additions are MIT licensed.
- Injection detector weights are NOT vendored: protectai/deberta-v3-base-prompt-injection-v2 at revision90c9989b1a342275dd0d1a95aad283c04e075671. Users download under the model repository's license/terms.

Source hashes are recorded in source_manifest.json. Vendored modules retain original authorship; local composition and native Gemini conversion are documented separately from defensive algorithms.
