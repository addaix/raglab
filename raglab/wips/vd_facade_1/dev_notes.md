Dev notes for vd_facade_1

- Implemented minimal facade modules: types, registry, components, interface, config
- Used simple in-memory store and simple embedder (length + unique chars)
- Kept design registry-based and dependency-light so tests run without external deps
- Next: expand embedders to wrap imbed when available, add langchain optional integration
