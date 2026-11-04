Feature: A person's own fact outranks a model's, and a refuted fact is marked, never erased

  @decided 2026-11-04: memory integrity. A fact that traces to the person's own
  words must not be displaced or overridden by anything a model wrote, and an
  attempt to do so stays visible as a contradiction instead of a quiet
  replacement. A fact can also be refuted: it closes without being deleted, and
  every fact derived from it is marked rather than dropped. Both rest on the
  provenance Engram already records (user, tool, external); no second trust
  system is added, and recall, scores and ranking are untouched.

  Every scenario names the test that holds it, in a comment on the line
  above it; scripts/features-are-bound.sh refuses a scenario with no test and
  a test name that no longer exists.

  # @test:test_a_models_extraction_cannot_close_a_persons_fact
  Scenario: A model's newer extraction does not close a fact that is the person's own
    Given a fact reflected from the person's own words, "Ivan works at Globex"
    When a later reflection over content that is not the person's extracts "Ivan works at Initech"
    Then the person's fact is still the one that was never closed
    And the model's fact is recorded beside it, so contradictions() lists the pair
    And one contradiction_found event names the model's fact and the person's fact
    And the reflection run does not count the conflict as resolved

  # @test:test_a_persons_newer_word_still_replaces_their_older_word
  Scenario: The person can still change their own mind
    Given a fact reflected from the person's own words, "Ivan works at Globex"
    When a later reflection over the person's own words extracts "Ivan works at Initech"
    Then the older fact is closed and the newer one is the only active value
    And the run counts one resolved contradiction

  # @test:test_reflect_contradiction_detection
  Scenario: A model's newer fact still replaces an older model fact
    Given a fact reflected from content that is not the person's
    When a later reflection extracts a different value for the same subject and predicate
    Then the older fact is closed and the newer one is the only active value

  # @test:test_a_quarantined_write_does_not_count_as_the_persons_word
  Scenario: A write made after reading untrusted content is not the person's word, on either side
    Given a person's fact and a later extraction from an episode labelled user but quarantined
    Then the person's fact is not closed
    And a fact that was itself extracted from a quarantined episode is not protected from a model

  # @test:test_the_protection_outlives_the_episodes_a_fact_was_extracted_from
  Scenario: Forgetting or pruning the source episodes does not remove the protection
    Given a fact reflected from the person's own words
    When the episode it came from is forgotten
    Then asking why still reports the person as its source
    And a later model extraction still cannot close it

  # @test:test_a_models_asserted_fact_leaves_a_persons_fact_in_place_and_is_reported
  Scenario: A fact asserted with a lower source leaves the person's fact in place and is reported
    Given a fact asserted as the person's own
    When another fact with a different value is asserted without that source
    Then both stay active, the person's one untouched
    And exactly one contradiction_found event names the new fact and the person's fact
    And an agreeing fact, or one from the person, raises no such event

  # @test:test_a_fact_that_cites_a_persons_episode_is_not_thereby_the_persons
  Scenario: Naming an episode as a fact's source does not borrow that episode's trust
    Given an episode the person wrote
    When a fact is asserted with that episode's id as its free-text source and no provenance
    Then asking why reports its source as unknown
    And it does not hold against a later conflicting fact

  # @test:test_refuting_a_fact_closes_it_and_keeps_it_in_the_history
  Scenario: Refuting a fact closes it and keeps it in the history
    Given an active fact
    When it is refuted
    Then it is no longer active and no longer a contradiction
    And timeline() still lists it with a superseded_at, who refuted it, and no replacement
    And an as_of date before the refutation still finds it, and a date after does not

  # @test:test_refuting_marks_every_fact_derived_from_it_and_closes_none_of_them
  Scenario: Refuting a fact marks what was derived from it, at any depth, and deletes nothing
    Given a fact, a second derived from it, a third derived from the second, and an unrelated fact
    When the first is refuted
    Then the second and third each carry the refuted fact's id as the reason they are doubtful
    And both are still active and nothing was deleted
    And the unrelated fact is not marked
    And why() on a marked fact returns the mark

  # @test:test_a_fact_derived_from_a_refuted_fact_is_born_marked
  Scenario: A fact built on a premise that is already refuted starts out marked
    Given a refuted fact
    When a new fact is asserted as derived from it, and another from that one
    Then both carry the refuted fact's id from the moment they are written

  # @test:test_refuting_twice_changes_nothing
  Scenario: Refuting twice is the same as refuting once
    Given a fact that was refuted and has a derived fact
    When it is refuted again
    Then its superseded_at is unchanged and nothing is marked twice

  # @test:test_a_models_refutation_of_a_persons_fact_is_refused
  Scenario: A model cannot refute the person's own fact
    Given a fact asserted as the person's own, and a fact derived from it
    When a refutation labelled tool, external, or user-but-quarantined is attempted
    Then it is refused and the fact stays active and unrefuted
    And the derived fact is not marked
    And the person's own refutation is accepted

  # @test:test_an_agent_cannot_refute_another_agents_fact
  Scenario: An agent refutes only its own facts
    Given a shared fact written by agent A and a private one
    When agent B tries to refute either
    Then the shared one is refused as not B's to close and the private one is not found
    And an instance with no agent identity, the file's operator, can refute

  # @test:test_the_cascade_reaches_a_private_fact_without_naming_it
  Scenario: The mark reaches another agent's private fact and does not name it
    Given agent A's shared fact and agent B's private fact derived from it
    When agent A refutes its fact
    Then agent A's result does not name B's private fact
    And B's fact carries the mark when B reads it

  # @test:test_a_cycle_or_a_diamond_in_derived_from_is_walked_once
  Scenario: A loop or a diamond among the derived facts does not stall or double the marking
    Given facts derived from a refuted one by two routes, and a loop between two of them
    When the first is refuted
    Then the walk ends, and each derived fact is marked exactly once

  # @test:test_refuting_a_missing_fact_raises
  Scenario: Refuting a fact that is not there is an error
    When an unknown fact id is refuted
    Then it raises KeyError

  # @test:test_a_refuted_fact_is_no_longer_a_contradiction
  Scenario: Refuting one side resolves a contradiction
    Given two active facts with different values for one subject and predicate
    When one is refuted
    Then contradictions() is empty

  # @test:test_asserting_a_fact_derived_from_an_unknown_fact_is_refused
  Scenario: A derivation must name facts that exist
    When a fact is asserted as derived from an id that is not a visible fact
    Then it raises KeyError and nothing is written

  # @test:test_the_memory_server_why_tool_returns_the_refutation_and_the_marks
  Scenario: The memory server's why tool returns the refutation and the marks
    Given a fact written through the memory server and a fact derived from it
    When the first is refuted and both are asked about through why
    Then the first reports who refuted it and the second reports the first as what it rests on

  # @test:test_the_cli_timeline_shows_a_refuted_fact_and_a_marked_one
  Scenario: The command line shows a refuted fact and a marked one
    Given a refuted fact, a fact derived from it, and an unrelated fact
    When the timeline of the entity is printed
    Then the refuted one says by whom, the derived one says what it rests on, and the unrelated one says neither

  # @test:test_the_async_wrapper_refutes_too
  Scenario: The async wrapper offers the same refutation
    Given an AsyncEngram with an active fact
    When the fact is refuted through it
    Then the fact is closed and marked as it would be through Engram

  # @test:test_a_store_from_before_the_trust_columns_opens_with_every_row_intact
  Scenario: A store from before this change opens, keeps every fact, and reads old facts as before
    Given a store stamped at the previous schema generation with facts in it
    When it is opened with this version
    Then the new columns exist and the schema marker is advanced
    And every fact is as it was, with no provenance recorded, no refutation and no marks
    And why() on an old fact reports what it reported before

  # @test:test_the_trust_columns_step_is_gated_on_the_schema_marker
  Scenario: The migration runs once and is safe to run again
    Given a store opened with this version
    When it is opened again and again
    Then the columns appear exactly once and the marker stays advanced
    And with the marker cleared the step runs, and with it set the step does not

  # @test:test_export_then_import_keeps_provenance_refutation_and_marks
  Scenario: Provenance, refutation and marks survive an export and an import
    Given a store with a person's fact, a refuted fact and a marked derived fact
    When it is exported and imported into a new store
    Then each fact keeps its provenance, who refuted it and its marks

  # @test:test_an_export_from_before_these_fields_imports_and_refuses_a_bad_source
  Scenario: An older export still imports, and a source outside the known ones is refused
    Given an export written before these fields existed
    When it is imported
    Then the facts arrive with no provenance, no refutation and no marks
    And an export naming a source outside user, tool, external and unknown is refused

  # @test:test_the_procedural_refusal_points_at_a_document_that_exists
  Scenario: The documents say what the code does about procedural memory
    Given a memory server asked to store a memory of the procedural kind
    When it refuses
    Then the message points at a document that exists in the repository
    And scripts/docs-match-code.sh holds the documents to the same statement
