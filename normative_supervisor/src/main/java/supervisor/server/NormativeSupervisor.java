package supervisor.server;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
//import java.util.logging.Logger;

import org.json.JSONArray;
import org.json.JSONObject;

import supervisor.games.Game;
import supervisor.normsys.NormBase;
import supervisor.games.Environment;
import util.ProjectUtils;

/**
 * The Normative Supervisor class initiates, updates, and fetches output for the module.
 * 
 * Should not be modified unless a bug is found; should not be extended unless absolutely necessary.
 * 
 * @author emery
 *
 */

public class NormativeSupervisor {
	protected Game game;
	protected int id;
	protected String type;
	protected String current;
	protected String reasonType;
	protected String normBaseType;
	protected String eval;
	protected ArrayList<String> possible;
	protected ArrayList<String> labels;
	protected ProjectUtils util = new ProjectUtils();
	protected String name;
	protected JSONObject state;
	protected ArrayList<String> stats;
	
	

	public NormativeSupervisor(String json) {
		current = json;
		JSONObject q = new JSONObject(json);
		possible = new ArrayList<String>();
		labels = new ArrayList<String>();
		if(q.has("game")) {
			state = q.getJSONObject("game");
		}
		eval = "";
		try {
		    name = q.getString("name");
		} catch(NullPointerException e){
			e.printStackTrace();
		}
    	reasonType = q.getString("reasoner");
    	normBaseType = q.getString("norms");
    	id = q.getInt("id");
    	game = createGame();
    	game.init();
    	stats = new ArrayList<String>();
    	/*String ts = new SimpleDateFormat("yyyy.MM.dd.HH.mm.ss").format(new Date());
    	File file = new File("//home//emery//stats-"+ts+".csv");
    	try {
    		FileWriter outputfile = new FileWriter(file);
            writer = new CSVWriter(outputfile);
            //String[] title = {"update-labels (ms)", "update-parse game(ms)", "reasoning (ms)", "theory size"};
            String[] title = {"update-labels (ms)", "update-parse game(ms)", "reasoning (ms)", "check ideal (ms)", "check sub-ideal (ms)", "theory size"};
            writer.writeNext(title);
		} catch (IOException e) {
			e.printStackTrace();
		}*/
	}            
    	

	public void update(String json) {
		stats.clear();
		current = json;
		JSONObject q = new JSONObject(json);
		type = q.getString("request");
		possible.clear();
		if(type.equals("FILTER")) {
			JSONArray p = q.getJSONArray("possible");
			for(int i = 0; i < p.length(); i++) {
	    		possible.add(p.getString(i));
	    	}
		}
		else if(type.equals("EVALUATION") || type.equals("METRIC")) {
			eval = q.getString("action");
			possible.add(eval);
		}
		else if(type.equals("DUAL-EVALUATION")) {
			eval = q.getString("action");
			JSONArray p = q.getJSONArray("possible");
			for(int i = 0; i < p.length(); i++) {
	    		possible.add(p.getString(i));
	    	}
		}
		int d = q.getInt("id");
		if(d != id) {
			id = d;
			game = createGame();
			game.init();
		}
		labels.clear();
		JSONArray l = q.getJSONArray("labels");
		long start1 = System.currentTimeMillis();
		for(int i = 0; i < l.length(); i++) {
    		labels.add(l.getString(i));
    	}
		long end1 = System.currentTimeMillis();
		String s1 = Long.toString(end1 - start1);
		stats.add(s1);
		long start2 = System.currentTimeMillis();
		game.update(parseGame(), possible);
		long end2 = System.currentTimeMillis();
		String s2 = Long.toString(end2 - start2);
		stats.add(s2);
	}
	
	
	public JSONObject fullfillRequest() {
		JSONObject response = new JSONObject();
		
		if(type.equals("FILTER")) {
			long start2 = System.currentTimeMillis();
            game.reason();
			long end2 = System.currentTimeMillis();
			String s2 = Long.toString(end2 - start2);
			stats.add(s2);
			String s3 = Integer.toString(game.getTheorySize());
			stats.add(s3);
            ArrayList<String> actions = game.findCompliantActions();
		        boolean compl = true;
		        if (actions.isEmpty()) {
		            System.out.println("No compliant actions. Locating maximally compliant action...");
		            actions = game.findBestNCActions();
		            printViolationFiles(possible, actions);
		            compl = false;
		          }
		        response = createFilterResponse(actions, compl);
		        actions.clear();
		}
		else if(type.equals("EVALUATION")) {
			long start2 = System.currentTimeMillis();
			game.reason();
			long end2 = System.currentTimeMillis();
			String s2 = Long.toString(end2 - start2);
			stats.add(s2);
			String s3 = Integer.toString(game.getTheorySize());
			stats.add(s3);
			boolean compl = game.checkAction(eval);
			response = createEvalResponse(compl);
		}
		else if(type.equals("METRIC")) {
			long start2 = System.currentTimeMillis();
			game.reason();
			long end2 = System.currentTimeMillis();
			String s2 = Long.toString(end2 - start2);
			stats.add(s2);
			String s3 = Integer.toString(game.getTheorySize());
			stats.add(s3);
			boolean compl = game.checkAction(eval);
			int score = game.scoreAction(eval);
			response = createMetricResponse(compl, score);
		}
		else if(type.equals("DUAL-EVALUATION")) {
			long start2 = System.currentTimeMillis();
			game.reason();
			long end2 = System.currentTimeMillis();
			String s2 = Long.toString(end2 - start2);
			stats.add(s2);
			long start3 = System.currentTimeMillis();
			boolean compl = game.checkAction(eval);
			long end3 = System.currentTimeMillis();
			String s3 = Long.toString(end3 - start3);
			stats.add(s3);
			long start4 = System.currentTimeMillis();
			boolean sub = game.checkActionSub(eval);
			long end4 = System.currentTimeMillis();
			String s4 = Long.toString(end4 - start4);
			stats.add(s4);
			String s5 = Integer.toString(game.getTheorySize());
			stats.add(s5);
			response = createDualEvalResponse(compl, sub);
		}
		//String[] st = new String[stats.size()];
		//st = stats.toArray(st);
		//writer.writeNext(st);
		return response;
	}
	
	
	public JSONObject createFilterResponse(List<String> actions, boolean compl){
		JSONObject response = new JSONObject();
		response.put("response", "RECOMMENDATION");
		JSONArray acts = new JSONArray(actions);
		response.put("actions", acts);
		response.put("compliant", compl);
    	return response;
    }
	
	
	public JSONObject createEvalResponse(boolean compl){
		JSONObject response = new JSONObject();
		response.put("response", "EVALUATION");
		response.put("compliant", compl);
    	return response;
    }
	
	public JSONObject createMetricResponse(boolean compl, int v){
		JSONObject response = new JSONObject();
		response.put("response", "METRIC");
		response.put("compliant", compl);
		response.put("violations", v);
    	return response;
    }
	
	public JSONObject createDualEvalResponse(boolean compl, boolean sub){
		JSONObject response = new JSONObject();
		response.put("response", "DUAL-EVALUATION");
		response.put("compliant", compl);
		response.put("sub-ideal", sub);
    	return response;
    }
    
    
    public void printViolationFiles(ArrayList<String> possible, ArrayList<String> non) {
    	DateFormat df = new SimpleDateFormat("yyMMddHHmmss");
		Date date = new Date();
		String datestr = df.format(date);
    	PrintWriter pw;
		try {
			pw = new PrintWriter("violation_"+datestr+".txt");
			pw.println("Game #: "+Integer.toString(id));
			pw.println("----- VIOLATION -----");
			pw.println("possible actions: "+possible.toString());
			pw.println("no compliant actions found");
			pw.println("minimally non-compliant: "+non.toString());
			pw.println("----- CONTEXT -----");
			pw.println(current);
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
    }
		
	
	public Environment parseGame() {
		Environment env = new Environment(0,0, util.getAllLabels(name));
		env.setLabels(labels);
		return env;
	}

	
	public Game createGame() {
		ArrayList<String> acts = util.getActionList(name);
		NormBase nb = util.defaultNormBase(name, normBaseType);
    	Game gm = new Game(parseGame(), nb, reasonType, acts);
    	return gm;
    }
	
	public Game getGame() {
		return game;
	}


}
