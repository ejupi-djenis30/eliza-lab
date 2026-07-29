//! A transparent, local dialogue and intent-classification laboratory.
//!
//! The deterministic rule engine remains available on its own. A trained model can be added for
//! intent routing, but input limits and the non-clinical safety exit always run before inference.

pub mod diagnostics;
pub mod ml;
pub mod open_set;
pub mod robustness;

pub const MAX_INPUT_CHARS: usize = 512;

const SAFETY_PHRASES: [&str; 14] = [
    "suicide",
    "suicidal",
    "kill myself",
    "end my life",
    "take my life",
    "hurt myself",
    "harm myself",
    "self harm",
    "want to die",
    "don't want to live",
    "do not want to live",
    "can't go on",
    "cannot go on",
    "immediate danger",
];

const FALLBACKS: [&str; 4] = [
    "Tell me a little more about that.",
    "What part of that stands out most to you?",
    "How did you arrive at that thought?",
    "What would change if you looked at it another way?",
];

/// Confidence information produced by the optional learned intent model.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelTrace {
    pub model_version: String,
    pub label: String,
    pub accepted: bool,
    pub confidence: f64,
    pub margin: f64,
    pub probabilities: std::collections::BTreeMap<String, f64>,
    pub top_features: Vec<ml::FeatureContribution>,
}

/// A response together with the rule trace shown by the learning interface.
#[derive(Debug, Clone, PartialEq)]
pub struct Reply {
    pub text: String,
    pub rule_id: &'static str,
    pub keyword: Option<String>,
    pub transformed_fragment: Option<String>,
    pub turn: usize,
    pub model_trace: Option<ModelTrace>,
}

/// Deterministic conversation engine with no transcript storage.
#[derive(Debug, Clone, Default)]
pub struct ElizaEngine {
    turn: usize,
}

impl ElizaEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn turn(&self) -> usize {
        self.turn
    }

    pub fn respond(&mut self, input: &str) -> Reply {
        self.turn = self.turn.saturating_add(1);
        let turn = self.turn;
        let trimmed = input.trim();

        if trimmed.is_empty() {
            return reply(
                "Take your time. What would you like to examine?",
                "empty-input",
                None,
                None,
                turn,
            );
        }

        if trimmed.chars().nth(MAX_INPUT_CHARS).is_some() {
            return reply(
                "That is more text than this small teaching demo can inspect at once. Try one short thought.",
                "input-boundary",
                None,
                None,
                turn,
            );
        }

        let normalized = normalize(trimmed);

        if SAFETY_PHRASES
            .iter()
            .any(|phrase| contains_phrase(&normalized, phrase))
        {
            return reply(
                "This demo cannot assess or support an emergency. If you might act on thoughts of suicide or self-harm, call your local emergency number now or reach a trusted person who can stay with you.",
                "safety-boundary",
                Some("matched safety phrase"),
                None,
                turn,
            );
        }

        if contains_word(&normalized, "hello")
            || contains_word(&normalized, "hi")
            || normalized == "hey"
        {
            return reply(
                "Hello. What is on your mind today?",
                "greeting",
                Some("hello"),
                None,
                turn,
            );
        }

        for prefix in ["i feel ", "i am ", "i'm "] {
            if let Some(fragment) = normalized.strip_prefix(prefix) {
                let reflected = reflect(fragment);
                return reply(
                    format!("What makes you feel {reflected}?"),
                    "feeling-reflection",
                    Some(prefix.trim()),
                    Some(reflected),
                    turn,
                );
            }
        }

        if let Some((_, reason)) = normalized.split_once(" because ") {
            let reflected = reflect(reason);
            return reply(
                format!("What makes {reflected} an important reason for you?"),
                "because-probe",
                Some("because"),
                Some(reflected),
                turn,
            );
        }

        if let Some(fragment) = normalized.strip_prefix("my ") {
            let reflected = reflect(fragment);
            return reply(
                format!("How does your {reflected} affect you?"),
                "ownership-reflection",
                Some("my"),
                Some(reflected),
                turn,
            );
        }

        if normalized.ends_with('?') {
            return reply(
                "What answer would feel most useful to explore?",
                "question-return",
                Some("question"),
                None,
                turn,
            );
        }

        reply(
            FALLBACKS[(turn - 1) % FALLBACKS.len()],
            "fallback",
            None,
            None,
            turn,
        )
    }

    /// Routes a prompt through a learned intent model after enforcing hard input and safety
    /// boundaries. Low-confidence predictions abstain and use a deterministic fallback.
    pub fn respond_with_model(&mut self, input: &str, model: &ml::IntentModel) -> Reply {
        self.turn = self.turn.saturating_add(1);
        let turn = self.turn;
        let trimmed = input.trim();

        if trimmed.is_empty() {
            return reply(
                "Take your time. What would you like to examine?",
                "empty-input",
                None,
                None,
                turn,
            );
        }
        if trimmed.chars().nth(MAX_INPUT_CHARS).is_some() {
            return reply(
                "That is more text than this small teaching demo can inspect at once. Try one short thought.",
                "input-boundary",
                None,
                None,
                turn,
            );
        }

        let normalized = normalize(trimmed);
        if SAFETY_PHRASES
            .iter()
            .any(|phrase| contains_phrase(&normalized, phrase))
        {
            return reply(
                "This demo cannot assess or support an emergency. If you might act on thoughts of suicide or self-harm, call your local emergency number now or reach a trusted person who can stay with you.",
                "safety-boundary",
                Some("matched safety phrase"),
                None,
                turn,
            );
        }

        let prediction = model.predict(trimmed);
        let response = match prediction.label.as_str() {
            "greeting" => Some("Hello. What would you like to examine today?"),
            "feeling" => Some("Which part of that feeling would be useful to examine?"),
            "reason" => Some("Which part of that explanation matters most here?"),
            "ownership" => Some("How does that situation affect what you can do next?"),
            "question" => Some("What answer would feel most useful to explore?"),
            "goal" => Some("What is the smallest concrete step you could test next?"),
            "observation" => Some("What part of that observation stands out most to you?"),
            _ => None,
        };
        let accepted = prediction.accepted && response.is_some();
        let model_trace = ModelTrace {
            model_version: model.model_version.clone(),
            label: prediction.label.clone(),
            accepted,
            confidence: prediction.confidence,
            margin: prediction.margin,
            probabilities: prediction.probabilities.clone(),
            top_features: prediction.top_features.clone(),
        };

        if accepted {
            Reply {
                text: response
                    .expect("accepted predictions have a mapped response")
                    .into(),
                rule_id: "ml-intent",
                keyword: Some(prediction.label),
                transformed_fragment: None,
                turn,
                model_trace: Some(model_trace),
            }
        } else {
            Reply {
                text: FALLBACKS[(turn - 1) % FALLBACKS.len()].into(),
                rule_id: "ml-abstain",
                keyword: Some(prediction.label),
                transformed_fragment: None,
                turn,
                model_trace: Some(model_trace),
            }
        }
    }

    /// Uses the primary open-set bundle while keeping the same hard input and safety boundaries as
    /// the legacy classifier. The operating policy, including abstention, is owned by the bundle.
    pub fn respond_with_open_set(
        &mut self,
        input: &str,
        runtime: &open_set::CompiledModel,
    ) -> Reply {
        self.turn = self.turn.saturating_add(1);
        let turn = self.turn;
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return reply(
                "Take your time. What would you like to examine?",
                "empty-input",
                None,
                None,
                turn,
            );
        }
        if trimmed.chars().nth(MAX_INPUT_CHARS).is_some() {
            return reply(
                "That is more text than this small teaching demo can inspect at once. Try one short thought.",
                "input-boundary",
                None,
                None,
                turn,
            );
        }
        let normalized = normalize(trimmed);
        if SAFETY_PHRASES
            .iter()
            .any(|phrase| contains_phrase(&normalized, phrase))
        {
            return reply(
                "This demo cannot assess or support an emergency. If you might act on thoughts of suicide or self-harm, call your local emergency number now or reach a trusted person who can stay with you.",
                "safety-boundary",
                Some("matched safety phrase"),
                None,
                turn,
            );
        }

        let prediction = runtime.predict(trimmed);
        let response = match prediction.label.as_str() {
            "greeting" => Some("Hello. What would you like to examine today?"),
            "feeling" => Some("Which part of that feeling would be useful to examine?"),
            "reason" => Some("Which part of that explanation matters most here?"),
            "ownership" => Some("How does that situation affect what you can do next?"),
            "question" => Some("What answer would feel most useful to explore?"),
            "goal" => Some("What is the smallest concrete step you could test next?"),
            "observation" => Some("What part of that observation stands out most to you?"),
            _ => None,
        };
        let accepted = prediction.accepted && response.is_some();
        let model_trace = ModelTrace {
            model_version: runtime.model().model_version.clone(),
            label: prediction.label.clone(),
            accepted,
            confidence: prediction.confidence,
            margin: prediction.probability_margin,
            probabilities: prediction.probabilities.clone(),
            top_features: prediction
                .explanation
                .top_contributions
                .iter()
                .map(|contribution| ml::FeatureContribution {
                    feature: contribution.feature.clone(),
                    value: contribution.value,
                    weight: contribution.top_weight - contribution.runner_up_weight,
                    contribution: contribution.contribution,
                })
                .collect(),
        };
        if accepted {
            Reply {
                text: response
                    .expect("accepted predictions have a mapped response")
                    .into(),
                rule_id: "ml-intent",
                keyword: Some(prediction.label),
                transformed_fragment: None,
                turn,
                model_trace: Some(model_trace),
            }
        } else {
            Reply {
                text: FALLBACKS[(turn - 1) % FALLBACKS.len()].into(),
                rule_id: "ml-abstain",
                keyword: Some(prediction.label),
                transformed_fragment: None,
                turn,
                model_trace: Some(model_trace),
            }
        }
    }
}

fn reply(
    text: impl Into<String>,
    rule_id: &'static str,
    keyword: Option<&str>,
    transformed_fragment: Option<String>,
    turn: usize,
) -> Reply {
    Reply {
        text: text.into(),
        rule_id,
        keyword: keyword.map(str::to_string),
        transformed_fragment,
        turn,
        model_trace: None,
    }
}

fn normalize(value: &str) -> String {
    let lowercase = value
        .chars()
        .flat_map(char::to_lowercase)
        .map(|character| match character {
            '’' | '‘' => '\'',
            _ => character,
        })
        .collect::<String>();
    lowercase.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn contains_word(haystack: &str, needle: &str) -> bool {
    words(haystack).iter().any(|word| word == needle)
}

fn contains_phrase(haystack: &str, needle: &str) -> bool {
    let haystack_words = words(haystack);
    let needle_words = words(needle);
    !needle_words.is_empty()
        && haystack_words
            .windows(needle_words.len())
            .any(|window| window == needle_words)
}

fn words(value: &str) -> Vec<String> {
    value
        .split(|character: char| !character.is_alphanumeric() && character != '\'')
        .map(|word| word.trim_matches('\''))
        .filter(|word| !word.is_empty())
        .map(str::to_string)
        .collect()
}

fn reflect(value: &str) -> String {
    words(value)
        .iter()
        .map(|word| match word.as_str() {
            "i" => "you",
            "me" => "you",
            "my" => "your",
            "mine" => "yours",
            "am" => "are",
            "you" => "I",
            "your" => "my",
            "yours" => "mine",
            "are" => "am",
            _ => word,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greets_without_storing_the_message() {
        let mut engine = ElizaEngine::new();
        let response = engine.respond("Hello");

        assert_eq!(response.rule_id, "greeting");
        assert_eq!(engine.turn(), 1);
        assert_eq!(
            std::mem::size_of::<ElizaEngine>(),
            std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn reflects_a_feeling() {
        let mut engine = ElizaEngine::new();
        let response = engine.respond("I feel uncertain about my next step.");

        assert_eq!(response.rule_id, "feeling-reflection");
        assert_eq!(
            response.transformed_fragment.as_deref(),
            Some("uncertain about your next step")
        );
        assert_eq!(
            response.text,
            "What makes you feel uncertain about your next step?"
        );
    }

    #[test]
    fn explains_a_because_rule() {
        let mut engine = ElizaEngine::new();
        let response = engine.respond("I paused because my plan changed");

        assert_eq!(response.rule_id, "because-probe");
        assert_eq!(response.keyword.as_deref(), Some("because"));
        assert!(response.text.contains("your plan changed"));
    }

    #[test]
    fn routes_urgent_language_to_a_clear_boundary() {
        let mut engine = ElizaEngine::new();
        let response = engine.respond("I might hurt myself");

        assert_eq!(response.rule_id, "safety-boundary");
        assert!(response.text.contains("emergency number"));
        assert!(!response.text.to_lowercase().contains("diagnos"));
    }

    #[test]
    fn safety_matching_uses_word_boundaries() {
        let mut engine = ElizaEngine::new();
        assert_ne!(
            engine.respond("I want to skill myself").rule_id,
            "safety-boundary"
        );

        let response = engine.respond("I don’t want to live");
        assert_eq!(response.rule_id, "safety-boundary");
        assert_eq!(response.keyword.as_deref(), Some("matched safety phrase"));
    }

    #[test]
    fn rejects_oversized_input() {
        let mut engine = ElizaEngine::new();
        let response = engine.respond(&"x".repeat(MAX_INPUT_CHARS + 1));

        assert_eq!(response.rule_id, "input-boundary");
    }

    #[test]
    fn cycles_deterministic_fallbacks() {
        let mut engine = ElizaEngine::new();
        let first = engine.respond("A statement");
        let second = engine.respond("Another statement");

        assert_eq!(first.rule_id, "fallback");
        assert_eq!(second.rule_id, "fallback");
        assert_ne!(first.text, second.text);
    }

    #[test]
    fn safety_boundary_precedes_learned_inference() {
        let dataset = ml::Dataset::from_tsv(
            "id\tlabel\ttext\n\
             g1\tgreeting\thello\n\
             g2\tgreeting\thi there\n\
             g3\tgreeting\tgood morning\n\
             g4\tgreeting\tgreetings\n\
             g5\tgreeting\thello again\n\
             q1\tquestion\twhat now?\n\
             q2\tquestion\twhere next?\n\
             q3\tquestion\thow so?\n\
             q4\tquestion\twhich one?\n\
             q5\tquestion\tcan it work?\n",
        )
        .unwrap();
        let (model, _) = ml::IntentModel::train(&dataset, ml::TrainingConfig::default()).unwrap();
        let mut engine = ElizaEngine::new();

        let response = engine.respond_with_model("hello, I want to die", &model);

        assert_eq!(response.rule_id, "safety-boundary");
        assert!(response.model_trace.is_none());
    }
}
