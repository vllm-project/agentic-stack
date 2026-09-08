//! Model input-modality capabilities shared by the Codex catalog handler, the gateway
//! configuration loader, and the harness launcher.
//!
//! Codex strips image content from a request when its local model catalog says the selected
//! model accepts text only, so the catalog served over HTTP and the isolated catalog written
//! by a launcher must advertise the same resolved modalities. This module owns that contract:
//! the validated modality set, the upstream metadata the gateway recognizes, the local
//! override table, and the resolution order between them.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Upstream capability string recognized as image input support.
const IMAGE_CAPABILITY: &str = "image";
/// Upstream capability string recognized as reasoning support.
const REASONING_CAPABILITY: &str = "reasoning";

/// One input modality a served model accepts.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    /// Text input, which every model served through the gateway accepts.
    Text,
    /// Image input.
    Image,
}

impl Modality {
    /// The wire representation of this modality.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
        }
    }
}

/// Rejected input-modality configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ModalityError {
    /// The list held no modalities.
    #[error("input_modalities must list at least one modality; expected [\"text\"] or [\"text\", \"image\"]")]
    Empty,
    /// The list repeated a modality.
    #[error("input_modalities lists \"{}\" more than once", .0.as_str())]
    Duplicate(Modality),
    /// The list omitted text.
    #[error("input_modalities must include \"text\"; a coding harness cannot use an image-only model")]
    MissingText,
}

/// The validated input modalities advertised for one served model.
///
/// A coding harness always sends instructions, history, and tool output as text, so text is
/// mandatory and the only usable combinations are text and text-with-image. Every other
/// combination is rejected while parsing, which keeps empty, duplicated, image-only, and
/// misordered states unrepresentable.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(try_from = "Vec<Modality>", into = "Vec<Modality>")]
pub enum InputModalities {
    /// Text input only: the conservative fallback when nothing advertises image support.
    #[default]
    Text,
    /// Text and image input.
    TextAndImage,
}

impl InputModalities {
    /// The modalities in the canonical wire order.
    #[must_use]
    pub const fn as_slice(self) -> &'static [Modality] {
        match self {
            Self::Text => &[Modality::Text],
            Self::TextAndImage => &[Modality::Text, Modality::Image],
        }
    }

    /// Whether `modality` is accepted.
    #[must_use]
    pub const fn contains(self, modality: Modality) -> bool {
        matches!(
            (self, modality),
            (Self::Text | Self::TextAndImage, Modality::Text) | (Self::TextAndImage, Modality::Image)
        )
    }

    /// Whether image input is accepted.
    #[must_use]
    pub const fn supports_image(self) -> bool {
        matches!(self, Self::TextAndImage)
    }

    /// Validate an unordered modality list.
    ///
    /// # Errors
    ///
    /// Returns [`ModalityError`] when the list is empty, repeats a modality, or omits text.
    pub fn try_new(values: &[Modality]) -> Result<Self, ModalityError> {
        let mut text = false;
        let mut image = false;
        for value in values {
            let seen = match value {
                Modality::Text => &mut text,
                Modality::Image => &mut image,
            };
            if *seen {
                return Err(ModalityError::Duplicate(*value));
            }
            *seen = true;
        }
        match (text, image) {
            (true, true) => Ok(Self::TextAndImage),
            (true, false) => Ok(Self::Text),
            (false, true) => Err(ModalityError::MissingText),
            (false, false) => Err(ModalityError::Empty),
        }
    }
}

impl TryFrom<Vec<Modality>> for InputModalities {
    type Error = ModalityError;

    fn try_from(values: Vec<Modality>) -> Result<Self, Self::Error> {
        Self::try_new(&values)
    }
}

impl From<InputModalities> for Vec<Modality> {
    fn from(modalities: InputModalities) -> Self {
        modalities.as_slice().to_vec()
    }
}

/// Capabilities an upstream `/v1/models` entry advertises.
///
/// Unrecognized capability strings are ignored: the upstream vocabulary is vendor-defined and
/// may grow, and an unknown string is never evidence of a capability.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UpstreamCapabilities {
    /// The entry advertises image input.
    pub image: bool,
    /// The entry advertises reasoning output.
    pub reasoning: bool,
}

impl UpstreamCapabilities {
    /// Recognize the capability strings an upstream model entry advertises.
    #[must_use]
    pub fn from_advertised<S: AsRef<str>>(entries: &[S]) -> Self {
        entries
            .iter()
            .fold(Self::default(), |capabilities, entry| match entry.as_ref() {
                IMAGE_CAPABILITY => Self {
                    image: true,
                    ..capabilities
                },
                REASONING_CAPABILITY => Self {
                    reasoning: true,
                    ..capabilities
                },
                _ => capabilities,
            })
    }
}

/// Local input-modality overrides, keyed by served model ID.
#[derive(Clone, Debug, Default)]
pub struct ModelCapabilities {
    overrides: BTreeMap<String, InputModalities>,
}

impl ModelCapabilities {
    /// Build a resolver from configured per-model overrides.
    #[must_use]
    pub fn new(overrides: BTreeMap<String, InputModalities>) -> Self {
        Self { overrides }
    }

    /// Resolve the input modalities advertised for `model_id`.
    ///
    /// Precedence is an explicit local override, then recognized upstream metadata, then the
    /// conservative text-only fallback. An explicit text-only override therefore wins over
    /// upstream image metadata, and a model with neither an override nor metadata stays
    /// text-only rather than being guessed from its name.
    #[must_use]
    pub fn resolve(&self, model_id: &str, upstream: UpstreamCapabilities) -> InputModalities {
        if let Some(modalities) = self.overrides.get(model_id) {
            return *modalities;
        }
        if upstream.image {
            InputModalities::TextAndImage
        } else {
            InputModalities::Text
        }
    }
}

/// The subset of the gateway's Codex model catalog a launcher needs.
///
/// Every other catalog field is ignored so that catalog growth cannot break a launcher.
#[derive(Debug, Default, Deserialize)]
pub struct CodexCatalogCapabilities {
    /// Catalog entries in the order the gateway advertises them.
    #[serde(default)]
    pub models: Vec<CodexModelCapabilities>,
}

/// One catalog entry's model identity and resolved input modalities.
#[derive(Debug, Deserialize)]
pub struct CodexModelCapabilities {
    /// Served model ID, matched exactly and case-sensitively.
    pub slug: String,
    /// The modalities the gateway resolved for this model.
    pub input_modalities: InputModalities,
}

impl CodexCatalogCapabilities {
    /// Select `model`, or the first advertised entry when no model was requested.
    #[must_use]
    pub fn select(&self, model: Option<&str>) -> Option<&CodexModelCapabilities> {
        match model {
            Some(model) => self.models.iter().find(|entry| entry.slug == model),
            None => self.models.first(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde::Deserialize;

    use super::{
        CodexCatalogCapabilities, InputModalities, Modality, ModalityError, ModelCapabilities, UpstreamCapabilities,
    };

    #[derive(Debug, Deserialize)]
    struct ModelSection {
        input_modalities: InputModalities,
    }

    #[test]
    fn validation_canonicalizes_accepted_modality_lists() {
        assert_eq!(InputModalities::try_new(&[Modality::Text]), Ok(InputModalities::Text));
        assert_eq!(
            InputModalities::try_new(&[Modality::Text, Modality::Image]),
            Ok(InputModalities::TextAndImage)
        );
        assert_eq!(
            InputModalities::try_new(&[Modality::Image, Modality::Text]),
            Ok(InputModalities::TextAndImage),
            "declaration order must not change the resolved modalities"
        );
    }

    #[test]
    fn validation_rejects_unusable_modality_lists() {
        assert_eq!(InputModalities::try_new(&[]), Err(ModalityError::Empty));
        assert_eq!(
            InputModalities::try_new(&[Modality::Image]),
            Err(ModalityError::MissingText)
        );
        assert_eq!(
            InputModalities::try_new(&[Modality::Text, Modality::Text]),
            Err(ModalityError::Duplicate(Modality::Text))
        );
        assert_eq!(
            InputModalities::try_new(&[Modality::Text, Modality::Image, Modality::Image]),
            Err(ModalityError::Duplicate(Modality::Image))
        );
    }

    #[test]
    fn modality_errors_name_the_wire_values() {
        assert!(ModalityError::Empty.to_string().contains("[\"text\", \"image\"]"));
        assert_eq!(
            ModalityError::Duplicate(Modality::Image).to_string(),
            "input_modalities lists \"image\" more than once"
        );
        assert!(ModalityError::MissingText.to_string().contains("must include \"text\""));
    }

    #[test]
    fn modalities_serialize_in_canonical_wire_order() {
        assert_eq!(
            serde_json::to_string(&InputModalities::Text).expect("serialize text"),
            "[\"text\"]"
        );
        assert_eq!(
            serde_json::to_string(&InputModalities::TextAndImage).expect("serialize text and image"),
            "[\"text\",\"image\"]"
        );
    }

    #[test]
    fn modalities_round_trip_through_json_and_toml() {
        let parsed: InputModalities = serde_json::from_str("[\"image\",\"text\"]").expect("parse JSON modalities");
        assert_eq!(parsed, InputModalities::TextAndImage);

        let section: ModelSection =
            toml::from_str("input_modalities = [\"text\", \"image\"]").expect("parse TOML modalities");
        assert_eq!(section.input_modalities, InputModalities::TextAndImage);
    }

    #[test]
    fn unknown_modality_names_the_accepted_values() {
        let error = toml::from_str::<ModelSection>("input_modalities = [\"video\"]")
            .expect_err("an unknown modality must be rejected");
        let message = error.to_string();

        assert!(message.contains("video"), "{message}");
        assert!(message.contains("text"), "{message}");
        assert!(message.contains("image"), "{message}");
    }

    #[test]
    fn invalid_modality_lists_are_rejected_while_parsing() {
        let error =
            toml::from_str::<ModelSection>("input_modalities = []").expect_err("an empty list must be rejected");
        assert!(error.to_string().contains("at least one modality"), "{error}");

        let error = toml::from_str::<ModelSection>("input_modalities = [\"image\"]")
            .expect_err("an image-only list must be rejected");
        assert!(error.to_string().contains("must include \"text\""), "{error}");

        let error = toml::from_str::<ModelSection>("input_modalities = [\"text\", \"text\"]")
            .expect_err("a duplicated modality must be rejected");
        assert!(error.to_string().contains("more than once"), "{error}");
    }

    #[test]
    fn modalities_report_their_members() {
        assert_eq!(InputModalities::Text.as_slice(), [Modality::Text]);
        assert_eq!(
            InputModalities::TextAndImage.as_slice(),
            [Modality::Text, Modality::Image]
        );
        assert!(InputModalities::Text.contains(Modality::Text));
        assert!(!InputModalities::Text.contains(Modality::Image));
        assert!(InputModalities::TextAndImage.contains(Modality::Image));
        assert!(!InputModalities::Text.supports_image());
        assert!(InputModalities::TextAndImage.supports_image());
    }

    #[test]
    fn upstream_capabilities_recognize_only_known_strings() {
        assert_eq!(
            UpstreamCapabilities::from_advertised(&["image", "reasoning"]),
            UpstreamCapabilities {
                image: true,
                reasoning: true
            }
        );
        assert_eq!(
            UpstreamCapabilities::from_advertised(&["vision", "multimodal", "IMAGE"]),
            UpstreamCapabilities::default(),
            "capability support must never be guessed from unrecognized strings"
        );
        assert_eq!(
            UpstreamCapabilities::from_advertised::<String>(&[]),
            UpstreamCapabilities::default()
        );
    }

    #[test]
    fn resolution_prefers_configuration_over_upstream_metadata() {
        let capabilities = ModelCapabilities::new(BTreeMap::from([
            ("vision-model".to_owned(), InputModalities::TextAndImage),
            ("pinned-text-model".to_owned(), InputModalities::Text),
        ]));
        let advertises_image = UpstreamCapabilities {
            image: true,
            reasoning: false,
        };
        let advertises_nothing = UpstreamCapabilities::default();

        assert_eq!(
            capabilities.resolve("vision-model", advertises_nothing),
            InputModalities::TextAndImage,
            "a configured vision model must advertise images without upstream metadata"
        );
        assert_eq!(
            capabilities.resolve("pinned-text-model", advertises_image),
            InputModalities::Text,
            "an explicit text-only override must win over upstream image metadata"
        );
        assert_eq!(
            capabilities.resolve("unconfigured-model", advertises_image),
            InputModalities::TextAndImage,
            "recognized upstream metadata must be used when no override exists"
        );
        assert_eq!(
            capabilities.resolve("unconfigured-model", advertises_nothing),
            InputModalities::Text,
            "an unknown model without metadata must stay text-only"
        );
        assert_eq!(
            ModelCapabilities::default().resolve("vision-model", advertises_nothing),
            InputModalities::Text
        );
    }

    #[test]
    fn catalog_selection_matches_slugs_exactly() {
        let catalog: CodexCatalogCapabilities = serde_json::from_str(
            r#"{"models":[
                {"slug":"first-model","input_modalities":["text"],"display_name":"ignored"},
                {"slug":"vision-model","input_modalities":["text","image"]}
            ]}"#,
        )
        .expect("parse catalog");

        assert_eq!(catalog.select(None).expect("first entry").slug, "first-model");
        let selected = catalog.select(Some("vision-model")).expect("selected entry");
        assert_eq!(selected.input_modalities, InputModalities::TextAndImage);
        assert!(
            catalog.select(Some("Vision-Model")).is_none(),
            "slugs are case-sensitive"
        );
        assert!(catalog.select(Some("missing-model")).is_none());
    }

    #[test]
    fn catalog_without_models_selects_nothing() {
        let catalog: CodexCatalogCapabilities = serde_json::from_str("{}").expect("parse empty catalog");

        assert!(catalog.models.is_empty());
        assert!(catalog.select(None).is_none());
        assert!(catalog.select(Some("any-model")).is_none());
    }
}
