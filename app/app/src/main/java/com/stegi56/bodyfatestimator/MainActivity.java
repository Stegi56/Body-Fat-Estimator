package com.stegi56.bodyfatestimator;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.AssetFileDescriptor;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.widget.EditText;
import android.widget.TextView;

import android.content.res.AssetManager;

import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Properties;


public class MainActivity extends AppCompatActivity {
    private EditText ankleField;
    private EditText kneeField;
    private EditText thighField;
    private EditText wristField;
    private EditText bicepField;
    private EditText waistField;
    private EditText forearmField;
    private EditText hipsField;
    private EditText heightField4;
    private EditText chestField;
    private EditText neckField;
    private EditText heightField;
    private EditText weightField;
    private TextView percentFat;

    private Interpreter tflite;
    private float[][] inputValues = new float[1][9];
    private float[][] outputValues = new float[1][1];
    private Properties properties = new Properties();
    private AssetManager assetManager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        assetManager = getAssets();

        initialiseFields();
        loadFields();
        initialiseListeners();

        try {
            tflite = new Interpreter(loadModelFile(assetManager, "BodyFatEstimator.tflite"));
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private void loadFields() {
        SharedPreferences preferences = getSharedPreferences("fields", Context.MODE_PRIVATE);

        // Load and set the values for each field
        heightField.setText(preferences.getString("height", ""));
        weightField.setText(preferences.getString("weight", ""));
        chestField.setText(preferences.getString("chest", ""));
        wristField.setText(preferences.getString("wrist", ""));
        waistField.setText(preferences.getString("waist", ""));
        forearmField.setText(preferences.getString("forearm", ""));
        hipsField.setText(preferences.getString("hips", ""));
        bicepField.setText(preferences.getString("bicep", ""));
        thighField.setText(preferences.getString("thigh", ""));
        kneeField.setText(preferences.getString("knee", ""));
        ankleField.setText(preferences.getString("ankle", ""));
        neckField.setText(preferences.getString("neck", ""));
    }

    private void storeFields() {
        SharedPreferences preferences = getSharedPreferences("fields", Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = preferences.edit();

        // Update file fields with app fields
        editor.putString("height", heightField.getText().toString());
        editor.putString("weight", weightField.getText().toString());
        editor.putString("chest", chestField.getText().toString());
        editor.putString("wrist", wristField.getText().toString());
        editor.putString("waist", waistField.getText().toString());
        editor.putString("forearm", forearmField.getText().toString());
        editor.putString("hips", hipsField.getText().toString());
        editor.putString("bicep", bicepField.getText().toString());
        editor.putString("thigh", thighField.getText().toString());
        editor.putString("knee", kneeField.getText().toString());
        editor.putString("ankle", ankleField.getText().toString());
        editor.putString("neck", neckField.getText().toString());

        // Save the changes
        editor.apply();
    }

    private void initialiseFields(){
        // initialise fields
        ankleField = findViewById(R.id.ankleField);
        kneeField = findViewById(R.id.kneeField);
        thighField = findViewById(R.id.thighField);
        wristField = findViewById(R.id.wristField);
        bicepField = findViewById(R.id.bicepField);
        waistField = findViewById(R.id.waistField);
        forearmField = findViewById(R.id.forearmField);
        hipsField = findViewById(R.id.hipsField);
        heightField4 = findViewById(R.id.waistField);
        chestField = findViewById(R.id.chestField);
        neckField = findViewById(R.id.neckField);
        heightField = findViewById(R.id.heightField);
        weightField = findViewById(R.id.weightField);
        percentFat = findViewById(R.id.percentFat);
    }

    private void initialiseListeners(){
        // add listeners to update fat% upon text changes
        ankleField.addTextChangedListener(textWatcher);
        kneeField.addTextChangedListener(textWatcher);
        thighField.addTextChangedListener(textWatcher);
        wristField.addTextChangedListener(textWatcher);
        bicepField.addTextChangedListener(textWatcher);
        waistField.addTextChangedListener(textWatcher);
        forearmField.addTextChangedListener(textWatcher);
        hipsField.addTextChangedListener(textWatcher);
        heightField4.addTextChangedListener(textWatcher);
        chestField.addTextChangedListener(textWatcher);
        neckField.addTextChangedListener(textWatcher);
        heightField.addTextChangedListener(textWatcher);
        weightField.addTextChangedListener(textWatcher);
    }

    private MappedByteBuffer loadModelFile(AssetManager assets, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelPath);
        FileInputStream fileInputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private TextWatcher textWatcher = new TextWatcher() {
        @Override
        public void beforeTextChanged(CharSequence charSequence, int start, int count, int after) {
        }

        @Override
        public void onTextChanged(CharSequence charSequence, int start, int before, int count) {
            updatePercentFat();
            storeFields();
        }

        @Override
        public void afterTextChanged(Editable editable) {
        }
    };

    private void updatePercentFat () {
        // If at least one input field is empty, don't update
        if (ankleField.getText().toString().isEmpty() ||
                kneeField.getText().toString().isEmpty() ||
                thighField.getText().toString().isEmpty() ||
                wristField.getText().toString().isEmpty() ||
                bicepField.getText().toString().isEmpty() ||
                waistField.getText().toString().isEmpty() ||
                forearmField.getText().toString().isEmpty() ||
                hipsField.getText().toString().isEmpty() ||
                heightField4.getText().toString().isEmpty() ||
                chestField.getText().toString().isEmpty() ||
                neckField.getText().toString().isEmpty() ||
                heightField.getText().toString().isEmpty() ||
                weightField.getText().toString().isEmpty()) {
            return;
        }

        // kg to lbs + cm to inch
        float weightLbs = Float.parseFloat(weightField.getText().toString()) * 2.2f;
        float heightInches = Float.parseFloat(heightField.getText().toString()) / 0.3937f;
        // bmi  = (weight * weight) / height
        inputValues[0][0] = ((weightLbs * weightLbs) / heightInches);
        inputValues[0][1] = Float.parseFloat(waistField.getText().toString()) / Float.parseFloat(neckField.getText().toString());
        inputValues[0][2] = Float.parseFloat(bicepField.getText().toString()) / Float.parseFloat(wristField.getText().toString());
        inputValues[0][3] = Float.parseFloat(waistField.getText().toString());
        inputValues[0][4] = Float.parseFloat(thighField.getText().toString()) / Float.parseFloat(ankleField.getText().toString());
        inputValues[0][5] = Float.parseFloat(chestField.getText().toString());
        inputValues[0][6] = Float.parseFloat(forearmField.getText().toString());
        inputValues[0][7] = Float.parseFloat(hipsField.getText().toString());
        inputValues[0][8] = Float.parseFloat(kneeField.getText().toString());

        float mean = 19.150793f;
        float variance = 69.757904f;

        // Normalize the inputValues array
        for (int i = 0; i < inputValues.length; i++) {
            for (int j = 0; j < inputValues[i].length; j++) {
                // Normalize each feature of the input sample
                inputValues[i][j] = (inputValues[i][j] - mean) / (float) Math.sqrt(variance);
            }
        }

        //run inputs into model
        tflite.run(inputValues, outputValues);
        float estimatedFatPercentage = outputValues[0][0];

        // Denormalise output (output * variance) + mean
        // vals taken from model pipeline
        estimatedFatPercentage = (estimatedFatPercentage * (float)Math.sqrt(variance)) + mean;
        percentFat.setText(String.format("%.2f", estimatedFatPercentage) + "%");
    }
}