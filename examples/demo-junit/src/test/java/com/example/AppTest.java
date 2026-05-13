package com.example;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;

class AppTest {
    private static final Path STATE = Path.of("build", "flaky-state.txt");

    private static int nextRunCount() throws IOException {
        int value = Files.exists(STATE) ? Integer.parseInt(Files.readString(STATE)) : 0;
        Files.createDirectories(STATE.getParent());
        Files.writeString(STATE, Integer.toString(value + 1));
        return value;
    }

    @Test
    void flakyToggle() throws IOException {
        assertEquals(0, nextRunCount() % 2);
    }

    @Test
    void stablePass() {
        assertEquals(2, 1 + 1);
    }
}
