# demo-junit

Small Ant/JUnit 5 project for exercising the FLAST JUnit provider.

The provider expects the JUnit Platform Console standalone jar at
`lib/junit-platform-console-standalone.jar`, or in `JUNIT_PLATFORM_CONSOLE_JAR`.

```bash
mkdir -p lib
curl -fsSL https://repo1.maven.org/maven2/org/junit/platform/junit-platform-console-standalone/1.10.3/junit-platform-console-standalone-1.10.3.jar \
  -o lib/junit-platform-console-standalone.jar

flast init junit
flast observe junit --runs 3
flast train
flast diagnostics junit --format json
```
