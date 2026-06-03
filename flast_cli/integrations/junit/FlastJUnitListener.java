package flast;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import org.junit.platform.engine.TestExecutionResult;
import org.junit.platform.engine.support.descriptor.MethodSource;
import org.junit.platform.launcher.TestExecutionListener;
import org.junit.platform.launcher.TestIdentifier;
import org.junit.platform.launcher.TestPlan;

public final class FlastJUnitListener implements TestExecutionListener {
    private final Path projectRoot = Path.of(System.getProperty("user.dir")).toAbsolutePath().normalize();
    private final String runId = UUID.randomUUID().toString();
    private final List<String> collected = new ArrayList<>();
    private final List<String> outcomes = new ArrayList<>();

    @Override
    public void testPlanExecutionStarted(TestPlan testPlan) {
        Path config = projectRoot.resolve(".flast").resolve("config.toml");
        if (!Files.exists(config)) {
            return;
        }
        for (TestIdentifier root : testPlan.getRoots()) {
            collectTests(testPlan, root);
        }
    }

    @Override
    public void executionFinished(TestIdentifier testIdentifier, TestExecutionResult testExecutionResult) {
        if (!testIdentifier.isTest()) {
            return;
        }
        testId(testIdentifier).ifPresent(testId -> outcomes.add(jsonOutcome(testId, outcome(testExecutionResult))));
    }

    @Override
    public void testPlanExecutionFinished(TestPlan testPlan) {
        Path config = projectRoot.resolve(".flast").resolve("config.toml");
        if (!Files.exists(config) || collected.isEmpty()) {
            return;
        }

        collected.sort(Comparator.naturalOrder());
        outcomes.sort(Comparator.naturalOrder());
        Path manifest = projectRoot.resolve(".flast").resolve("runs").resolve(runId).resolve("manifest.json");
        try {
            Files.createDirectories(manifest.getParent());
            Files.writeString(manifest, manifestJson(), StandardCharsets.UTF_8);
        } catch (IOException error) {
            throw new RuntimeException("Failed to write FLAST JUnit manifest", error);
        }
    }

    private void collectTests(TestPlan testPlan, TestIdentifier identifier) {
        if (identifier.isTest()) {
            testId(identifier).ifPresent(testId -> collected.add("{\"test_id\":\"" + escape(testId) + "\"}"));
        }
        for (TestIdentifier child : testPlan.getChildren(identifier)) {
            collectTests(testPlan, child);
        }
    }

    private Optional<String> testId(TestIdentifier identifier) {
        return identifier.getSource()
            .filter(MethodSource.class::isInstance)
            .map(MethodSource.class::cast)
            .map(source -> filePart(source) + "::" + source.getClassName() + "::" + source.getMethodName());
    }

    private String filePart(MethodSource source) {
        Path file = projectRoot.resolve("src/test/java")
            .resolve(source.getClassName().replace('.', '/') + ".java")
            .normalize();
        return projectRoot.relativize(file).toString().replace('\\', '/');
    }

    private String outcome(TestExecutionResult result) {
        switch (result.getStatus()) {
            case SUCCESSFUL:
                return "passed";
            case ABORTED:
                return "skipped";
            case FAILED:
                return "failed";
            default:
                return "error";
        }
    }

    private String jsonOutcome(String testId, String outcome) {
        return "{\"test_id\":\"" + escape(testId) + "\",\"outcome\":\"" + outcome + "\"}";
    }

    private String manifestJson() {
        return "{\n"
            + "  \"schema_version\": 1,\n"
            + "  \"framework\": \"junit\",\n"
            + "  \"run_id\": \"" + runId + "\",\n"
            + "  \"project_root\": \"" + escape(projectRoot.toString()) + "\",\n"
            + "  \"tool\": {\"name\": \"junit-platform\", \"version\": \"unknown\"},\n"
            + "  \"collected\": [" + String.join(",", collected) + "],\n"
            + "  \"outcomes\": [" + String.join(",", outcomes) + "]\n"
            + "}\n";
    }

    private String escape(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
